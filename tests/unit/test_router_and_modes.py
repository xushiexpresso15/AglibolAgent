"""Unit tests for IntentRouter, dynamic AgentMode transitions, approval gate, and workspace commands."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aglibol.agents.router import IntentRouter
from aglibol.cli.interactive.commands import SlashCommandHandler
from aglibol.cli.interactive.repl import InteractiveSession
from aglibol.core.types import AgentMode


def test_intent_router_greetings_and_questions():
    # Direct greetings
    assert IntentRouter.classify("hi") == AgentMode.CHAT
    assert IntentRouter.classify("Hello!") == AgentMode.CHAT
    assert IntentRouter.classify("hey there") == AgentMode.CHAT
    assert IntentRouter.classify("greetings") == AgentMode.CHAT

    # Conversational questions
    assert IntentRouter.classify("who are you?") == AgentMode.CHAT
    assert IntentRouter.classify("what can you do") == AgentMode.CHAT
    assert IntentRouter.classify("introduce yourself") == AgentMode.CHAT
    assert IntentRouter.classify("explain how closures work") == AgentMode.CHAT


def test_intent_router_planning_and_code():
    # Planning requests
    assert IntentRouter.classify("make a plan for this project") == AgentMode.PLANNER
    assert IntentRouter.classify("plan first before coding") == AgentMode.PLANNER
    assert IntentRouter.classify("architecture design for a web scraper") == AgentMode.PLANNER

    # Direct coding requests
    assert IntentRouter.classify("write a python function to compute primes") == AgentMode.CODER
    assert IntentRouter.classify("implement quicksort algorithm") == AgentMode.CODER
    assert IntentRouter.classify("fix bug in main.py") == AgentMode.CODER

    # Review requests
    assert IntentRouter.classify("code review on src/app.py") == AgentMode.REVIEWER
    assert IntentRouter.classify("check this code for syntax errors") == AgentMode.REVIEWER


def test_intent_router_explicit_switch_overrides_current_mode():
    # Even if current mode is CODER, explicit request switches to PLANNER
    assert IntentRouter.classify("plan first", current_mode=AgentMode.CODER) == AgentMode.PLANNER
    assert IntentRouter.classify("make a plan", current_mode=AgentMode.CODER) == AgentMode.PLANNER

    # If no explicit request and in CODER, stay in CODER
    assert IntentRouter.classify("add a docstring", current_mode=AgentMode.CODER) == AgentMode.CODER


def test_intent_router_plan_approval_detection():
    assert IntentRouter.detect_plan_approval("yes") == "approve"
    assert IntentRouter.detect_plan_approval("y") == "approve"
    assert IntentRouter.detect_plan_approval("approve") == "approve"
    assert IntentRouter.detect_plan_approval("proceed") == "approve"
    assert IntentRouter.detect_plan_approval("looks good, start") == "approve"

    assert IntentRouter.detect_plan_approval("no") == "reject"
    assert IntentRouter.detect_plan_approval("n") == "reject"
    assert IntentRouter.detect_plan_approval("cancel") == "reject"
    assert IntentRouter.detect_plan_approval("stop") == "reject"

    assert IntentRouter.detect_plan_approval("change database to SQLite") == "feedback"
    assert IntentRouter.detect_plan_approval("add unit tests in step 2") == "feedback"


@pytest.mark.asyncio
async def test_workspace_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "main.py").write_text("print('hello')\n")
        (root / "subdir").mkdir()

        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        # 1. /workspace status
        assert await handler.handle("/workspace") is True

        # 2. /workspace list
        assert await handler.handle("/workspace list") is True

        # 3. /workspace new subproj
        assert await handler.handle("/workspace new subproj") is True
        assert session.workspace_root == (root / "subproj").resolve()
        assert session.workspace_root.is_dir()

        # 4. /workspace switch back to root
        assert await handler.handle(f"/workspace {root}") is True
        assert session.workspace_root == root.resolve()


def test_home_directory_workspace_protection():
    # If workspace_root is home directory, safeguard to aglibol_workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir).resolve()
        with patch("pathlib.Path.home", return_value=fake_home):
            session = InteractiveSession(workspace_root=fake_home)
            assert session.workspace_root == fake_home / "aglibol_workspace"


def test_intent_router_writer_classification():
    # Writing and documentation queries
    assert IntentRouter.classify("write a README.md") == AgentMode.WRITER
    assert IntentRouter.classify("draft technical documentation") == AgentMode.WRITER
    assert IntentRouter.classify("write a tutorial article") == AgentMode.WRITER
    assert IntentRouter.classify("write docs for this project") == AgentMode.WRITER
    assert IntentRouter.classify("draft readme for github") == AgentMode.WRITER
    assert IntentRouter.classify("switch to writer mode") == AgentMode.WRITER

    # Explicit override in another mode
    assert IntentRouter.classify("write docs", current_mode=AgentMode.CODER) == AgentMode.WRITER


def test_agent_registry_includes_writer():
    from aglibol.agents.registry import AgentRegistry
    from aglibol.agents.writer import WriterAgent

    reg = AgentRegistry()
    assert "writer" in reg.list_agents()
    assert reg.get("writer") == WriterAgent


@pytest.mark.asyncio
async def test_writer_agent_execution_and_extraction():
    from aglibol.agents.writer import WriterAgent

    writer = WriterAgent()
    assert writer.name == "writer"
    assert writer.role == "writer"

    # Test doc extraction
    text = (
        "Here is the documentation:\n"
        "```markdown:README.md\n"
        "# My Project\n\nWelcome to the docs.\n"
        "```\n"
    )
    extracted = writer._extract_all_doc_blocks(text)
    assert "README.md" in extracted
    assert "# My Project" in extracted["README.md"]


@pytest.mark.asyncio
async def test_model_command_writer_and_all_bindings():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session = InteractiveSession(workspace_root=root)
        handler = SlashCommandHandler(session)

        # 1. /mode writer
        assert await handler.handle("/mode writer") is True
        assert session.agent_mode == AgentMode.WRITER

        # 2. /model writer my-writer-model:latest
        assert await handler.handle("/model writer my-writer-model:latest") is True
        assert session.model_overrides["writer"] == "my-writer-model:latest"

        # 3. /model chat my-chat-model:latest
        assert await handler.handle("/model chat my-chat-model:latest") is True
        assert session.model_overrides["chat"] == "my-chat-model:latest"

        # 4. /model all unified-model:latest
        assert await handler.handle("/model all unified-model:latest") is True
        assert session.model_overrides["chat"] == "unified-model:latest"
        assert session.model_overrides["planner"] == "unified-model:latest"
        assert session.model_overrides["coder"] == "unified-model:latest"
        assert session.model_overrides["reviewer"] == "unified-model:latest"
        assert session.model_overrides["writer"] == "unified-model:latest"


def test_multilingual_intent_router_chinese_inputs():
    # Chinese planning request with explicit switch
    zh_plan = "\u8acb\u4f60\u8f49\u5230planning mod \u7136\u5f8cplan\u4e00\u500b\u8a08\u7b97\u8cea\u6578\u7684\u5c0fpython\u7a0b\u5f0f"
    assert IntentRouter.classify(zh_plan) == AgentMode.PLANNER

    # Chinese coding request
    zh_code = "\u8a08\u7b97\u8cea\u6578\u7684python\u7a0b\u5f0f"
    assert IntentRouter.classify(zh_code) == AgentMode.CODER

    # Chinese review request
    zh_review = "\u5be9\u67e5\u9019\u6bb5\u4ee3\u7801\u662f\u5426\u6709bug"
    assert IntentRouter.classify(zh_review) == AgentMode.REVIEWER

    # Chinese writer request
    zh_writer = "\u5beb\u4e00\u7bc7\u6587\u7ae0\u4ecb\u7d39\u9019\u500b\u9805\u76ee"
    assert IntentRouter.classify(zh_writer) == AgentMode.WRITER

    # Chinese greeting
    zh_greeting = "\u4f60\u597d"
    assert IntentRouter.classify(zh_greeting) == AgentMode.CHAT


def test_multilingual_plan_approval_chinese():
    assert IntentRouter.detect_plan_approval("\u597d\u554a") == "approve"
    assert IntentRouter.detect_plan_approval("\u597d\u7684") == "approve"
    assert IntentRouter.detect_plan_approval("\u53ef\u4ee5") == "approve"
    assert IntentRouter.detect_plan_approval("\u6c92\u554f\u984c") == "approve"

    assert IntentRouter.detect_plan_approval("\u4e0d\u8981") == "reject"
    assert IntentRouter.detect_plan_approval("\u53d6\u6d88") == "reject"


def test_responsive_ui_rendering_components():
    from aglibol.cli.interactive.tui_renderer import TUIRenderer

    # 1. Thought box (2-line limit)
    multi_line_thought = (
        "<thinking>\nStep 1: Parse AST\nStep 2: Detect primes\nStep 3: Output result\n</thinking>"
    )
    TUIRenderer.render_thought_box(multi_line_thought, agent_name="Coder")
    TUIRenderer.render_thought_card(multi_line_thought, agent_name="Coder")

    # 2. Tool box (2-line limit)
    TUIRenderer.render_tool_box(
        "write_file", details={"path": "primes.py", "lines": 42}, status="writing"
    )
    TUIRenderer.render_tool_start("write_file", args={"path": "primes.py"})
    TUIRenderer.render_tool_finish("write_file", success=True, output="File saved successfully")

    # 3. Mode switch confirmation
    approved = TUIRenderer.confirm_mode_switch(
        current_mode="chat",
        current_model="gemma4:e4b",
        target_mode="planner",
        target_model="orinth",
        reason="Planning requested",
        auto_approve=True,
    )
    assert approved is True


def test_context_compression_with_episodic_offload():
    from aglibol.core.context import ContextManager
    from aglibol.core.types import ChatMessage
    from aglibol.memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "episodic.jsonl"
        episodic = EpisodicMemory(log_file)

        messages = [
            ChatMessage(role="system", content="System instruction"),
            ChatMessage(role="user", content="Initial goal"),
        ]
        for _ in range(10):
            messages.append(ChatMessage(role="tool", content="x" * 300, tool_name="shell"))
        messages.append(ChatMessage(role="assistant", content="Recent response"))
        messages.append(ChatMessage(role="user", content="Next turn"))

        compressed = ContextManager.compress_messages(
            messages=messages,
            max_context_tokens=300,
            headroom_ratio=0.60,
            episodic_memory=episodic,
        )

        assert len(compressed) < len(messages)
        turns = episodic.read_all_turns()
        assert len(turns) > 0
        assert any(t.get("extra", {}).get("offloaded_to_disk") for t in turns)


def test_intent_router_typo_resilience_and_goal_cleaning():
    # 1. User typo 'plaaan mod' with repeated vowels
    prompt = "ok can you change to plaaan mod and plaaan a small program that can tell the difference between prime numbers?"
    assert IntentRouter.detect_explicit_mode_request(prompt) == AgentMode.PLANNER
    assert IntentRouter.classify(prompt) == AgentMode.PLANNER

    # 2. Extract clean objective without mode switch command prefix
    clean = IntentRouter.extract_clean_goal(prompt)
    assert "change to plaaan mod" not in clean
    assert "prime numbers" in clean

    # 3. Resilient variations across other modes
    assert IntentRouter.classify("swiiitch to ccooode mode") == AgentMode.CODER
    assert IntentRouter.classify("change to reviiiew mode and check syntax") == AgentMode.REVIEWER
    assert IntentRouter.classify("switch to wrrriter mode") == AgentMode.WRITER
    assert IntentRouter.classify("can you enter chhhhat mode") == AgentMode.CHAT
