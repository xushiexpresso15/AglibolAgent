"""Writer agent for technical documentation, articles, and copywriting."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from aglibol.agents.base import BaseAgent
from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.scheduler import ModelScheduler
from aglibol.core.types import AgentState, ChatMessage
from aglibol.ollama.client import OllamaClient
from aglibol.tools.registry import ToolRegistry

logger = logging.getLogger("aglibol.agents.writer")


class WriterAgent(BaseAgent):
    """Specialized in drafting READMEs, technical documentation, articles, and copywriting."""

    name = "writer"
    role = "writer"
    default_model = "qwen2.5:7b"

    system_prompt = """You are a Master Author, Essayist, and Principal Technical Writer.
Your job is to produce comprehensive, lucid, engaging, and beautifully structured writing across essays, prose, articles, technical documentation, stories, and copywriting.

Operational Guidelines:
1. Always analyze user requirements, tone, language, and length constraints, and structure your outline, key themes, and narrative arc inside <thinking>...</thinking> tags first, before drafting your writing.
2. If writing prose or an essay (e.g. 1000-word prose/essay), write a rich, complete, high-quality, and deeply expressive piece of writing without truncation or placeholder summaries.
3. Structure your work clearly using standard markdown (headings, paragraphs, quotes).
4. Persist your work to disk using the `write_file` tool (e.g. 'essay.md', 'article.md', 'README.md') or output the complete text directly.
5. Provide accurate, polished, production-ready writing without placeholder tags.
"""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(
            model_name=model_name,
            tools=["read_file", "write_file", "list_dir", "search_replace"],
        )

    async def execute(
        self,
        state: AgentState,
        client: OllamaClient,
        scheduler: ModelScheduler,
        tool_registry: ToolRegistry,
        event_bus: EventBus | None = None,
    ) -> AgentState:
        workspace = state.workspace_dir or os.getcwd()
        prompt = (
            f"Writing Objective:\n{state.user_goal}\n\n"
            f"Active Workspace Directory: {workspace}\n\n"
            "Please research existing files if needed and draft the requested writing, prose, or article. "
            "Persist any target documents to disk using write_file or provide complete markdown text."
        )

        messages = [ChatMessage(role="user", content=prompt)]

        resp = await self._run_agent_loop(
            messages=messages,
            client=client,
            scheduler=scheduler,
            tool_registry=tool_registry,
            event_bus=event_bus,
        )

        extracted_blocks: dict[str, str] = {}
        written_files = resp.raw_response.get("written_files", {})
        for filepath, content in written_files.items():
            if filepath and content:
                if "\n" not in content and "\\n" in content:
                    content = content.replace("\\n", "\n").replace("\\t", "\t")
                fname = Path(filepath).name
                extracted_blocks[fname] = content

        # Extract markdown file blocks
        extracted_blocks.update(
            self._extract_all_doc_blocks(resp.content, context_hint=state.user_goal)
        )

        # Fallback: detect pseudo write_file in resp.content
        m_pseudo = re.search(
            r'write_file\s*\(\s*["\']([^"\']+\.md)["\']\s*,\s*"""([\s\S]*?)"""', resp.content
        )
        if m_pseudo:
            extracted_blocks[Path(m_pseudo.group(1)).name] = m_pseudo.group(2).strip()

        # Fallback: If no markdown fence was used, or extracted doc is too short compared to response
        goal_lower = state.user_goal.lower()
        if any(k in goal_lower for k in ("readme", "readme.md")):
            doc_name = "README.md"
        elif any(k in goal_lower for k in ("essay", "prose", "\u6563\u6587", "\u4f5c\u6587")):
            doc_name = "essay.md"
        elif any(k in goal_lower for k in ("story", "\u5c0f\u8aaa", "\u6545\u4e8b")):
            doc_name = "story.md"
        else:
            doc_name = "document.md"

        existing_doc = extracted_blocks.get(doc_name, "")
        if (not extracted_blocks or len(existing_doc) < 100) and len(resp.content.strip()) > len(
            existing_doc
        ):
            extracted_blocks[doc_name] = resp.content.strip()

        for filename, doc_content in extracted_blocks.items():
            if not filename:
                filename = f"document_step_{state.current_step}_{len(state.artifacts) + 1}.md"
            state.artifacts[filename] = doc_content

            if state.workspace_dir:
                try:
                    target_path = Path(state.workspace_dir) / filename
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(doc_content, encoding="utf-8")
                except Exception as e:
                    logger.warning("Could not auto-persist doc %s to workspace: %s", filename, e)

        # Append assistant response so interactive views and histories can display it
        state.messages.append(ChatMessage(role="assistant", content=resp.content))
        state.status = "success"
        state.is_completed = True

        if event_bus:
            await event_bus.emit(
                AgentEvent(
                    event_type="DOC_GENERATED",
                    session_id=state.session_id,
                    step=state.current_step,
                    agent=self.name,
                    data={
                        "length": len(resp.content),
                        "artifacts": list(state.artifacts.keys()),
                    },
                )
            )

        return state

    @staticmethod
    def _extract_all_doc_blocks(text: str, context_hint: str = "") -> dict[str, str]:
        """Extract markdown documents from fences with filename hints."""
        blocks: dict[str, str] = {}
        pattern = re.compile(r"```(?:(\w+))?(?::([^\n\r]+))?\s*\n([\s\S]*?)```")

        # Skip typical code languages that are not document contents
        non_doc_langs = {
            "python",
            "py",
            "bash",
            "sh",
            "zsh",
            "cmd",
            "bat",
            "powershell",
            "ps1",
            "javascript",
            "js",
            "typescript",
            "ts",
            "json",
            "yaml",
            "yml",
            "toml",
            "c",
            "cpp",
            "csharp",
            "cs",
            "java",
            "go",
            "rust",
            "rs",
            "sql",
            "html",
            "css",
        }

        for match in pattern.finditer(text):
            _lang = (match.group(1) or "").strip().lower()
            if _lang in non_doc_langs:
                continue

            filename_header = (match.group(2) or "").strip()
            content = match.group(3).strip()
            if not content:
                continue

            filename = filename_header
            if not filename:
                for line in content.splitlines()[:5]:
                    stripped = line.strip()
                    m = re.match(r"^[#\s]*[Ff]ile:\s*([^\s]+)", stripped)
                    if m:
                        filename = m.group(1).strip()
                        break

            if not filename and context_hint:
                m_save = re.search(
                    r"(?:save\s+(?:as|to)|named)\s+([a-zA-Z0-9_\-\./\\]+\.md)\b",
                    context_hint,
                    re.IGNORECASE,
                )
                if m_save:
                    filename = Path(m_save.group(1)).name
                else:
                    goal_lower = context_hint.lower()
                    if any(k in goal_lower for k in ("readme", "readme.md")):
                        filename = "README.md"
                    elif any(
                        k in goal_lower for k in ("essay", "prose", "\u6563\u6587", "\u4f5c\u6587")
                    ):
                        filename = "essay.md"
                    elif any(k in goal_lower for k in ("story", "\u5c0f\u8aaa", "\u6545\u4e8b")):
                        filename = "story.md"

            if not filename:
                filename = "document.md"

            if filename:
                # Keep longest content if multiple blocks target the same filename
                if filename not in blocks or len(content) > len(blocks[filename]):
                    blocks[filename] = content

        return blocks
