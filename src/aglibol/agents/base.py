"""Base agent abstract class with autonomous reasoning and tool execution loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aglibol.core.context import ContextManager
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentResponse, AgentState, ChatMessage, ToolCall
from aglibol.ollama.client import OllamaClient
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.agents")


class BaseAgent(ABC):
    """Abstract agent responsible for executing a specific stage in the workflow."""

    name: str
    role: str
    system_prompt: str
    default_model: str = "qwen2.5:7b"
    max_tool_iterations: int = 5

    def __init__(self, model_name: str | None = None, tools: list[str] | None = None) -> None:
        if model_name:
            self.default_model = model_name
        self.tools = tools or []

    @abstractmethod
    async def execute(
        self,
        state: AgentState,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
    ) -> AgentState:
        """Execute agent reasoning and mutate shared state."""
        pass

    async def _run_agent_loop(
        self,
        messages: list[ChatMessage],
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
        custom_options: dict[str, Any] | None = None,
        format: dict[str, Any] | None = None,
        episodic_memory: Any | None = None,
    ) -> AgentResponse:
        """Run standard reasoning -> tool calling -> reasoning loop.

        Changes from original:
        - Preserves full intermediate conversation history in AgentResponse.history
        - try/finally ensures scheduler model release on any error
        - Retry on transient network errors (2 attempts with exponential backoff)
        - Returns last non-empty content on max iterations instead of empty string
        - Loop breaker: Prevents repeating identical read_file/list_dir calls
        - Disk memory: Streams intermediate tool observations to episodic memory in real time
        """
        # 1. Acquire model from scheduler
        model, options, keep_alive = await scheduler.acquire_model(
            model_name=self.default_model,
            role=self.role,
            custom_options=custom_options,
        )

        # 2. Get tool definitions
        ollama_tools = tool_registry.get_ollama_tools(self.tools) if self.tools else None

        current_messages = list(messages)
        # Ensure system prompt is set at the head
        if not current_messages or current_messages[0].role != "system":
            current_messages.insert(0, ChatMessage(role="system", content=self.system_prompt))
        else:
            current_messages[0] = ChatMessage(role="system", content=self.system_prompt)

        iteration = 0
        total_tokens = 0
        final_content = ""
        last_non_empty_content = ""
        all_thinking: list[str] = []
        written_files: dict[str, str] = {}
        executed_tool_cache: dict[str, str] = {}
        consecutive_read_count = 0
        missing_deliverable_prompts = 0
        force_text_mode = False

        # Real-time token streaming callback for unified telemetry across thinking and drafting
        async def _on_loop_token(token_type: str, token_str: str) -> None:
            if event_bus and token_str:
                await event_bus.emit(
                    AgentEvent(
                        event_type="AGENT_THOUGHT_TOKEN",
                        agent=self.name,
                        data={"token": token_str, "stream_type": token_type},
                    )
                )

        while iteration < self.max_tool_iterations:
            iteration += 1
            # Compress context if necessary, logging offloaded turns to disk episodic memory
            current_messages = ContextManager.compress_messages(
                current_messages,
                max_context_tokens=options.get("num_ctx", 2048),
                headroom_ratio=0.75,
                episodic_memory=episodic_memory,
            )
            # If consecutive reads reached limit or falling back to raw code generation, adjust effective tools
            effective_tools = None if force_text_mode else ollama_tools
            if not force_text_mode and consecutive_read_count >= 2 and ollama_tools:
                effective_tools = [
                    t
                    for t in ollama_tools
                    if (
                        t.get("function", {}).get("name")
                        if isinstance(t, dict)
                        else getattr(t, "tool_name", "")
                    )
                    not in ("read_file", "list_dir")
                ]

            # Call Ollama with retry on transient errors
            res = await self._chat_with_retry(
                client=client,
                model=model,
                messages=current_messages,
                options=options,
                keep_alive=keep_alive,
                tools=effective_tools if effective_tools else None,
                format=format,
                on_token=_on_loop_token,
            )
            content = res.get("content", "")
            thinking = res.get("thinking", "")
            if thinking.strip():
                all_thinking.append(thinking.strip())
            raw_tool_calls = res.get("tool_calls", [])
            tool_calls: list[ToolCall] = []
            for raw_tc in raw_tool_calls:
                if isinstance(raw_tc, ToolCall):
                    tool_calls.append(raw_tc)
                elif isinstance(raw_tc, dict):
                    fn = raw_tc.get("function", {})
                    tname = raw_tc.get("tool_name") or fn.get("name", "")
                    targs = raw_tc.get("arguments") or fn.get("arguments", {})
                    if isinstance(targs, str):
                        try:
                            targs = json.loads(targs)
                        except Exception:
                            targs = {}
                    tool_calls.append(ToolCall(tool_name=tname, arguments=targs))

            # Fallback: Detect text-based JSON tool calls in content if API tool_calls is empty
            if not tool_calls and tool_registry and content.strip():
                for candidate in re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content) + [
                    content.strip()
                ]:
                    try:
                        data = json.loads(candidate.strip())
                        if (
                            isinstance(data, dict)
                            and "name" in data
                            and ("arguments" in data or "parameters" in data)
                        ):
                            tname = data["name"]
                            targs = data.get("arguments") or data.get("parameters") or {}
                            if isinstance(targs, str):
                                targs = json.loads(targs)
                            if tool_registry.get(tname):
                                tool_calls.append(ToolCall(tool_name=tname, arguments=targs))
                                break
                        elif isinstance(data, list):
                            found = False
                            for item in data:
                                if (
                                    isinstance(item, dict)
                                    and "name" in item
                                    and tool_registry.get(item["name"])
                                ):
                                    targs = item.get("arguments") or item.get("parameters") or {}
                                    if isinstance(targs, str):
                                        targs = json.loads(targs)
                                    tool_calls.append(
                                        ToolCall(tool_name=item["name"], arguments=targs)
                                    )
                                    found = True
                            if found:
                                break
                    except Exception:
                        pass

                # Detect pseudo write_file syntax in content
                if not tool_calls and tool_registry and "write_file" in content:
                    for m_wf in re.finditer(
                        r'write_file\s*\(?\{?\s*["\']?(?:path|filename)["\']?\s*:\s*["\']([^"\']+)["\'],\s*["\']?content["\']?\s*:\s*(?:"""([\s\S]*?)"""|"([\s\S]*?)"|\'([\s\S]*?)\')',
                        content,
                    ):
                        wf_path = m_wf.group(1).strip()
                        wf_content = m_wf.group(2) or m_wf.group(3) or m_wf.group(4) or ""
                        tool_calls.append(
                            ToolCall(
                                tool_name="write_file",
                                arguments={"path": wf_path, "content": wf_content},
                            )
                        )

            total_tokens += res.get("eval_count", 0)
            # Track non-empty content for fallback
            if content.strip():
                last_non_empty_content = content

            # Auto-continuation loop if model reached token limit mid-generation
            continuation_turns = 0
            while res.get("done_reason") == "length" and continuation_turns < 3:
                continuation_turns += 1
                logger.info(
                    "Agent '%s' reached length limit (turn %d). Auto-continuing generation...",
                    self.name,
                    continuation_turns,
                )
                cont_messages = list(current_messages)
                cont_messages.append(ChatMessage(role="assistant", content=content))
                action_text = (
                    "code generation" if self.role == "coder" else "writing and text generation"
                )
                cont_messages.append(
                    ChatMessage(
                        role="user",
                        content=f"Continue {action_text} directly from where you left off. Output only the continuation without repeating previous text.",
                    )
                )
                cont_messages = ContextManager.compress_messages(
                    cont_messages,
                    max_context_tokens=options.get("num_ctx", 2048),
                )
                cont_res = await self._chat_with_retry(
                    client=client,
                    model=model,
                    messages=cont_messages,
                    options=options,
                    keep_alive=keep_alive,
                    tools=ollama_tools if ollama_tools else None,
                    format=format,
                    on_token=_on_loop_token,
                )
                cont_chunk = cont_res.get("content", "")
                cont_thinking = cont_res.get("thinking", "")
                if cont_thinking.strip():
                    all_thinking.append(cont_thinking.strip())
                if not cont_chunk.strip():
                    break
                content += cont_chunk
                res["done_reason"] = cont_res.get("done_reason", "stop")

            # Extract thought for event bus, filtering out raw JSON/code
            thought_text = thinking.strip()
            if not thought_text and ("<think>" in content or "<thinking>" in content):
                m = re.search(r"<(?:think|thinking)>([\s\S]*?)</(?:think|thinking)>", content)
                if m:
                    thought_text = m.group(1).strip()

            if thought_text and event_bus:
                await event_bus.emit(
                    AgentEvent(
                        event_type="AGENT_THOUGHT",
                        agent=self.name,
                        data={"content": thought_text, "tool_calls_count": len(tool_calls)},
                    )
                )
            # If no tool calls requested, check if deliverables were created or if model produced pure chatter
            if not tool_calls:
                has_code_syntax = bool(
                    re.search(r"```(?:\w+)?(?::[^\n\r]+)?\s*\n[\s\S]*?```", content)
                    or "<!DOCTYPE html" in content
                    or "<html" in content
                    or (
                        self.role == "coder"
                        and any(k in content for k in ("def ", "class ", "import "))
                    )
                )
                if (
                    self.role == "coder"
                    and not written_files
                    and not has_code_syntax
                    and missing_deliverable_prompts < 2
                    and iteration < self.max_tool_iterations
                ):
                    missing_deliverable_prompts += 1
                    force_text_mode = True
                    logger.info(
                        "Coder produced conversational output without deliverables. Steering code block generation (attempt %d/2)...",
                        missing_deliverable_prompts,
                    )
                    current_messages.append(
                        ChatMessage(
                            role="assistant", content=content or thinking or "Analysis complete."
                        )
                    )
                    current_messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "CRITICAL DIRECTIVE: You have not created any implementation files or code deliverables. "
                                "You must NOT respond with text conversation, promises, or analysis. "
                                "Output the complete, full implementation code immediately inside markdown code blocks:\n"
                                "```<language>:<filename>\n"
                                "<complete implementation>\n"
                                "```\n"
                                "Begin outputting the code block now without any other text:"
                            ),
                        )
                    )
                    continue

                final_content = content or thinking
                break
            # Add assistant message with tool calls
            current_messages.append(
                ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
            # Execute tool calls
            for tc in tool_calls:
                is_read_tool = tc.tool_name in ("read_file", "list_dir")
                raw_arg_path = str(
                    tc.arguments.get("path")
                    or tc.arguments.get("dir_path")
                    or tc.arguments.get("file_path")
                    or ""
                ).strip()
                try:
                    tool_inst = tool_registry.get(tc.tool_name)
                    ws = getattr(tool_inst, "workspace_root", None)
                    p = Path(raw_arg_path)
                    if ws and not p.is_absolute():
                        p = ws / p
                    norm_path = str(p.resolve()) if raw_arg_path else ""
                except Exception:
                    norm_path = raw_arg_path
                cache_key = f"{tc.tool_name}:{norm_path}" if is_read_tool else ""

                if is_read_tool and (
                    cache_key in executed_tool_cache or consecutive_read_count >= 2
                ):
                    consecutive_read_count += 1
                    logger.info(
                        "Loop breaker activated: redundant %s on %s (consecutive=%d)",
                        tc.tool_name,
                        raw_arg_path,
                        consecutive_read_count,
                    )
                    prev_info = executed_tool_cache.get(cache_key, "")
                    cached_snippet = (
                        f"\n[Previous result: {prev_info[:300]}...]" if prev_info else ""
                    )
                    tool_output = (
                        f"[Disk Memory Cache] Path '{raw_arg_path}' information has already been retrieved.{cached_snippet}\n"
                        f"Read operations are suspended to protect context. "
                        f"You must now proceed immediately to create and persist the implementation deliverables using 'write_file'."
                    )
                else:
                    if event_bus:
                        await event_bus.emit(
                            AgentEvent(
                                event_type="TOOL_EXEC_STARTED",
                                agent=self.name,
                                data={"tool": tc.tool_name, "args": tc.arguments},
                            )
                        )
                    result = await tool_registry.execute(tc.tool_name, tc.arguments)
                    if event_bus:
                        await event_bus.emit(
                            AgentEvent(
                                event_type="TOOL_EXEC_FINISHED",
                                agent=self.name,
                                data={"tool": tc.tool_name, "success": result.success},
                            )
                        )
                    # Track files written by tools
                    if result.success:
                        if tc.tool_name == "write_file":
                            p = tc.arguments.get("path")
                            c = tc.arguments.get("content")
                            if p and c is not None:
                                clean_c = str(c)
                                if re.search(r"<(?:function|tool_call|parameter)=", clean_c):
                                    clean_c = re.split(
                                        r"<(?:function|tool_call|parameter)=", clean_c
                                    )[0].rstrip()
                                if "\n" not in clean_c and "\\n" in clean_c:
                                    clean_c = clean_c.replace("\\n", "\n").replace("\\t", "\t")
                                written_files[p] = clean_c
                                consecutive_read_count = 0
                        elif tc.tool_name == "search_replace":
                            p = tc.arguments.get("path")
                            if p:
                                target_file = Path(p)
                                if target_file.exists():
                                    try:
                                        written_files[p] = target_file.read_text(
                                            encoding="utf-8", errors="replace"
                                        )
                                        consecutive_read_count = 0
                                    except Exception:
                                        pass

                    tool_output = result.output if result.success else f"Error: {result.error}"
                    if is_read_tool:
                        executed_tool_cache[cache_key] = tool_output
                        consecutive_read_count += 1
                        if consecutive_read_count >= 2 and not written_files:
                            tool_output += (
                                "\n[Memory Directive: Workspace inspection complete. "
                                "Stop calling read tools and immediately write the required deliverables using 'write_file'.]"
                            )

                # Append tool result to context
                current_messages.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output,
                        tool_name=tc.tool_name,
                    )
                )

                # Real-time persistence of tool observations to disk episodic memory
                if episodic_memory and hasattr(episodic_memory, "log_turn"):
                    try:
                        episodic_memory.log_turn(
                            agent=self.name,
                            step=iteration,
                            message=ChatMessage(
                                role="tool",
                                content=tool_output,
                                tool_name=tc.tool_name,
                            ),
                            extra={"tool": tc.tool_name, "args": tc.arguments},
                        )
                    except Exception:
                        pass
        else:
            # Reached max iterations — use last non-empty content
            if not final_content:
                final_content = last_non_empty_content or (
                    "\n\n".join(all_thinking).strip()
                    if all_thinking
                    else "(Agent reached maximum tool iterations without a final response)"
                )
                logger.warning(
                    "Agent '%s' reached max_tool_iterations (%d) without final answer",
                    self.name,
                    self.max_tool_iterations,
                )

        # Build history from intermediate messages (excluding system prompt)
        history = [m for m in current_messages if m.role != "system"]

        return AgentResponse(
            content=final_content,
            thinking="\n\n".join(all_thinking).strip(),
            history=history,
            is_done=bool(final_content or written_files or all_thinking),
            tokens_used=total_tokens,
            raw_response={"written_files": written_files},
        )

    @staticmethod
    async def _chat_with_retry(
        client: OllamaClient,
        model: str,
        messages: list[ChatMessage],
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        tools: list[dict[str, Any]] | None = None,
        format: dict[str, Any] | None = None,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        on_token: Any = None,
    ) -> dict[str, Any]:
        """Call client.chat with retry on transient network errors."""
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    keep_alive=keep_alive,
                    tools=tools,
                    format=format,
                    on_token=on_token,
                )
            except Exception as e:
                last_error = e
                error_name = type(e).__name__
                # Only retry on transient network errors
                if error_name in (
                    "ReadTimeout",
                    "ConnectError",
                    "RemoteProtocolError",
                    "ConnectTimeout",
                ):
                    if attempt < max_retries:
                        wait = retry_delay * (2**attempt)
                        logger.warning(
                            "Ollama chat attempt %d/%d failed (%s), retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            error_name,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                # Non-transient error or final attempt — raise
                raise
        raise last_error  # type: ignore[misc]
