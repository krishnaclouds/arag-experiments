from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

from ..config import AgentConfig, compute_cost, get_api_key
from ..tools.chunk_read import ChunkReadTool
from ..tools.keyword_search import KeywordSearchTool
from ..tools.semantic_search import SemanticSearchTool
from .prompts import SUMMARIZATION_SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT


class AgentLoop:
    """
    ReAct-style A-RAG agent loop backed by the Anthropic tool-use API.

    At each step the model either:
      - calls one of the three retrieval tools (action → observation → repeat), or
      - emits a plain-text final answer (stopping the loop).

    If max_loops is reached without a final answer, the model is prompted
    for its best-effort answer given what it has gathered so far.
    """

    def __init__(
        self,
        config: AgentConfig,
        keyword_tool: KeywordSearchTool,
        semantic_tool: SemanticSearchTool,
        chunk_tool: ChunkReadTool,
        api_key: str | None = None,
    ) -> None:
        self.config = config
        self._tools: dict[str, Any] = {
            keyword_tool.name: keyword_tool,
            semantic_tool.name: semantic_tool,
            chunk_tool.name: chunk_tool,
        }
        self._chunk_tool = chunk_tool
        self._tool_schemas = [t.schema for t in self._tools.values()]
        self._client = anthropic.Anthropic(api_key=api_key or get_api_key())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict[str, Any]:
        """
        Run the agent loop for a single question.

        Returns a dict with:
          answer            : str   — final answer text
          trace             : list  — [{loop, tool, input, output}, ...]
          loops             : int   — number of iterations used
          input_tokens      : int
          output_tokens     : int
          cost_usd          : float — estimated API cost for this question
          latency_ms        : int   — wall-clock time from first call to final answer
          word_count        : int   — word count of the final answer
          max_loops_reached : bool (only present if True)
        """
        self._chunk_tool.reset()

        if self.config.task_type == "summarization":
            system_prompt = SUMMARIZATION_SYSTEM_PROMPT_TEMPLATE.format(
                target_length=self.config.target_length,
            )
        else:
            system_prompt = SYSTEM_PROMPT
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[dict] = []
        total_in = total_out = 0
        t_start = time.monotonic()

        for loop_idx in range(self.config.max_loops):
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                tools=self._tool_schemas,
                messages=messages,
            )
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            # ---- token budget guard ----
            if total_in + total_out > self.config.max_token_budget:
                answer = "".join(
                    b.text for b in response.content if hasattr(b, "text")
                ).strip()
                if answer:
                    return {
                        "answer": answer,
                        "trace": trace,
                        "loops": loop_idx + 1,
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "cost_usd": compute_cost(self.config.model, total_in, total_out),
                        "latency_ms": int((time.monotonic() - t_start) * 1000),
                        "word_count": len(answer.split()),
                        "max_loops_reached": True,
                    }
                break  # fall through to best-effort answer prompt

            # ---- final answer ----
            if response.stop_reason == "end_turn":
                answer = "".join(
                    b.text for b in response.content if hasattr(b, "text")
                ).strip()
                return {
                    "answer": answer,
                    "trace": trace,
                    "loops": loop_idx + 1,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cost_usd": compute_cost(self.config.model, total_in, total_out),
                    "latency_ms": int((time.monotonic() - t_start) * 1000),
                    "word_count": len(answer.split()),
                }

            # ---- tool call(s) ----
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool = self._tools.get(block.name)
                    if tool is None:
                        result: Any = {"error": f"Unknown tool: {block.name}"}
                    else:
                        try:
                            result = tool.run(**block.input)
                        except Exception as exc:
                            result = {"error": str(exc)}

                    trace.append(
                        {
                            "loop": loop_idx + 1,
                            "tool": block.name,
                            "input": block.input,
                            "output": result,
                        }
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

        # ---- max loops reached: ask for best-effort answer ----
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of retrieval steps. "
                    "Please provide your best answer based on what you have gathered so far."
                ),
            }
        )
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=messages,
        )
        answer = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens
        return {
            "answer": answer,
            "trace": trace,
            "loops": self.config.max_loops,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": compute_cost(self.config.model, total_in, total_out),
            "latency_ms": int((time.monotonic() - t_start) * 1000),
            "word_count": len(answer.split()),
            "max_loops_reached": True,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    dataset: str = "docfinqa",
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Run the agent for a single question using a dataset config YAML.

    Example:
        from src.arag import run_agent
        result = run_agent("What was Apple's gross margin in FY2023?", dataset="financebench")
        print(result["answer"])
    """
    if config_path is None:
        config_path = f"configs/{dataset}.yaml"

    config = AgentConfig.from_yaml(config_path)

    keyword_tool = KeywordSearchTool(chunks_file=config.chunks_file)
    semantic_tool = SemanticSearchTool(
        chunks_file=config.chunks_file,
        index_dir=config.index_dir,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
    )
    chunk_tool = ChunkReadTool(chunks_file=config.chunks_file)

    agent = AgentLoop(
        config=config,
        keyword_tool=keyword_tool,
        semantic_tool=semantic_tool,
        chunk_tool=chunk_tool,
    )
    return agent.run(question)
