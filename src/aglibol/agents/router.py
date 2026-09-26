"""Intent router and dynamic mode detector for Aglibol Agent."""

from __future__ import annotations

import re

from aglibol.core.types import AgentMode


class IntentRouter:
    """Classifies user input to determine the optimal AgentMode and dynamic transitions."""

    GREETINGS_PATTERN = re.compile(
        r"^(hi|hello|hey|yo|greetings|hola|\u4f60\u597d|\u60a8\u597d|\u55e8|\u54c8\u56c9)[\s!?.~]*$",
        re.IGNORECASE,
    )

    CONVERSATIONAL_QUESTIONS = [
        "who are you",
        "what can you do",
        "introduce yourself",
        "help me",
        "thank you",
        "thanks",
        "bye",
        "goodbye",
        "\u4f60\u662f\u8ab0",
        "\u4f60\u662f\u8c01",
        "\u80fd\u505a\u4ec0\u9ebc",
        "\u80fd\u505a\u4ec0\u4e48",
        "\u4ecb\u7d39\u4e00\u4e0b",
        "\u4ecb\u7ecd\u4e00\u4e0b",
        "\u8b1d\u8b1d",
        "\u8c22\u8c22",
    ]

    SWITCH_INDICATORS = [
        "switch to",
        "change to",
        "turn to",
        "enter",
        "mode",
        "mod",
        "\u8f49\u5230",
        "\u8f6c\u5230",
        "\u5207\u63db",
        "\u5207\u6362",
        "\u6362\u5230",
        "\u6539\u70ba",
        "\u6539\u4e3a",
        "\u9032\u5165",
        "\u8fdb\u5165",
    ]

    PLAN_KEYWORDS = [
        "plan first",
        "make a plan",
        "create a plan",
        "plan this",
        "architecture design",
        "step by step plan",
        "system architecture",
        "requirements analysis",
        "design architecture",
        "plan the project",
        "planning mod",
        "planning mode",
        "planner mode",
        "plan a",
        "plan an",
        "\u898f\u5283",
        "\u89c4\u5212",
        "\u67b6\u69cb\u8a2d\u8a08",
        "\u67b6\u6784\u8bbe\u8ba1",
        "\u8a2d\u8a08\u67b6\u69cb",
        "\u8bbe\u8ba1\u67b6\u6784",
        "\u5236\u5b9a\u8a08\u756b",
        "\u8a08\u756b",
        "\u8ba1\u5212",
        "\u898f\u5283\u6a21\u5f0f",
        "\u89c4\u5212\u6a21\u5f0f",
    ]

    REVIEW_KEYWORDS = [
        "code review",
        "audit code",
        "check for bugs",
        "review code",
        "syntax check",
        "review this",
        "find bugs",
        "find errors",
        "check this code",
        "security audit",
        "review mode",
        "reviewer mode",
        "\u5be9\u67e5",
        "\u5ba1\u67e5",
        "\u8a55\u5be9",
        "\u5be9\u67e5\u4ee3\u7801",
        "\u5ba1\u67e5\u4ee3\u7801",
        "\u8a55\u5be9\u4ee3\u7801",
        "\u4ee3\u7801\u8a55\u5be9",
        "\u4ee3\u7801\u5be9\u67e5",
        "\u7a0b\u5f0f\u78bc\u5be9\u67e5",
        "\u4ee3\u7801\u5ba1\u67e5",
        "\u6aa2\u67e5\u4ee3\u7801",
        "\u6aa2\u67e5\u7a0b\u5f0f",
        "\u68c0\u67e5\u4ee3\u7801",
        "\u6aa2\u67e5\u8a9e\u6cd5",
        "\u5be9\u67e5\u6a21\u5f0f",
        "\u5ba1\u67e5\u6a21\u5f0f",
        "\u627e\u51fa\u932f\u8aa4",
        "\u627e\u8775",
    ]

    WRITER_KEYWORDS = [
        "write doc",
        "write docs",
        "documentation",
        "write readme",
        "draft readme",
        "write article",
        "article",
        "tutorial",
        "blog post",
        "copywriting",
        "api doc",
        "user manual",
        "guide",
        "readme.md",
        "technical documentation",
        "write guide",
        "release notes",
        "readme",
        "writer mode",
        "write prose",
        "prose",
        "write essay",
        "essay",
        "write a story",
        "short story",
        "story",
        "creative writing",
        "write article",
        "draft article",
        "write a blog",
        "write poem",
        "poetry",
        "\u6587\u7ae0",
        "\u6587\u6a94",
        "\u6587\u6863",
        "\u8aaa\u660e\u6587\u4ef6",
        "\u8bf4\u660e\u6587\u4ef6",
        "\u5beb\u6587\u7ae0",
        "\u5199\u6587\u7ae0",
        "\u4f5c\u8005\u6a21\u5f0f",
        "\u64b0\u5beb\u6587\u4ef6",
        "\u6559\u7a0b",
        "\u6559\u5b78",
        "\u624b\u518a",
        "\u6307\u5357",
        "\u6587\u6848",
        "\u6563\u6587",
        "\u5beb\u6563\u6587",
        "\u5199\u6563\u6587",
        "\u4f5c\u6587",
        "\u5beb\u4f5c\u6587",
        "\u5199\u4f5c\u6587",
        "\u5c0f\u8aaa",
        "\u5c0f\u8bf4",
        "\u6545\u4e8b",
        "\u5beb\u6545\u4e8b",
        "\u5199\u6545\u4e8b",
        "\u77ed\u6587",
        "\u96a8\u7b46",
        "\u968f\u7b14",
    ]

    CODE_ACTION_KEYWORDS = [
        "create file",
        "write code",
        "implement",
        "generate code",
        "fix bug",
        "modify code",
        "refactor",
        "create a script",
        "write function",
        "implement algorithm",
        "build app",
        "write a python",
        "write a function",
        "write a script",
        "write a class",
        "write an algorithm",
        "coding mode",
        "coder mode",
        "python program",
        "python script",
        "python code",
        "html",
        "html page",
        "web page",
        "website",
        "landing page",
        "frontend",
        "front-end",
        "css",
        "javascript",
        "\u5beb\u7a0b\u5f0f",
        "\u5199\u4ee3\u7801",
        "\u5beb\u4ee3\u7801",
        "\u7de8\u7a0b",
        "\u7f16\u7a0b",
        "\u7de8\u5beb",
        "\u7f16\u5199",
        "\u5be6\u73fe",
        "\u5b9e\u73b0",
        "\u5beb\u4e00\u500b",
        "\u5199\u4e00\u4e2a",
        "\u8173\u672c",
        "\u51fd\u6578",
        "\u51fd\u5f0f",
        "\u7b97\u6cd5",
        "\u4fee\u5fa9bug",
        "\u91cd\u69cb",
        "\u7a0b\u5f0f",
        "\u4ee3\u7801",
        "\u7db2\u9801",
        "\u7f51\u9875",
        "\u505a\u7db2\u9801",
        "\u505a\u7f51\u9875",
        "\u5beb\u7db2\u9801",
        "\u5199\u7f51\u9875",
        "\u524d\u7aef",
        "\u7db2\u7ad9",
        "\u7f51\u7ad9",
        "\u505a\u7db2\u7ad9",
        "\u505a\u7f51\u7ad9",
        "\u5beb\u7db2\u7ad9",
        "\u5199\u7f51\u7ad9",
    ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse repeated character runs and normalize common typos in mode commands."""
        # 1. Collapse 3 or more repeated characters down to 1 (e.g. plaaan -> plan, swiiitch -> switch)
        collapsed = re.sub(r"(.)\1{2,}", r"\1", text.lower())
        # 2. Normalize common variations
        collapsed = re.sub(r"\bp+l+a+n+[a-z]*\b", "plan", collapsed)
        collapsed = re.sub(r"\bm+o+d+e*\b", "mode", collapsed)
        collapsed = re.sub(r"\bc+o+d+[a-z]*\b", "code", collapsed)
        collapsed = re.sub(r"\br+e+v+i+e+w+[a-z]*\b", "review", collapsed)
        collapsed = re.sub(r"\bw+r+i+t+[a-z]*\b", "write", collapsed)
        return collapsed

    @classmethod
    def detect_explicit_mode_request(cls, text: str) -> AgentMode | None:
        """Detect if the user explicitly requested a specific mode in their input text with typo resilience."""
        stripped = text.strip()
        lower = stripped.lower()
        norm = cls.normalize_text(stripped)

        # 1. Planner checks
        if any(kw in lower for kw in cls.PLAN_KEYWORDS) or any(
            kw in norm for kw in cls.PLAN_KEYWORDS
        ):
            return AgentMode.PLANNER
        if re.search(
            r"\b(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?plan\b", norm
        ):
            return AgentMode.PLANNER
        if re.search(r"\bplan\s+mode\b", norm):
            return AgentMode.PLANNER
        if any(s in lower for s in cls.SWITCH_INDICATORS) and any(
            k in lower for k in ("planning", "planner", "plan", "\u898f\u5283", "\u89c4\u5212")
        ):
            return AgentMode.PLANNER

        # 2. Reviewer checks
        if any(kw in lower for kw in cls.REVIEW_KEYWORDS) or any(
            kw in norm for kw in cls.REVIEW_KEYWORDS
        ):
            return AgentMode.REVIEWER
        if re.search(
            r"\b(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?review\b", norm
        ):
            return AgentMode.REVIEWER
        if re.search(r"\breview\s+mode\b", norm):
            return AgentMode.REVIEWER
        if any(s in lower for s in cls.SWITCH_INDICATORS) and any(
            k in lower
            for k in (
                "review",
                "reviewer",
                "audit",
                "\u5be9\u67e5",
                "\u5ba1\u67e5",
                "\u6aa2\u67e5",
                "\u68c0\u67e5",
            )
        ):
            return AgentMode.REVIEWER

        # 3. Writer checks
        if any(kw in lower for kw in cls.WRITER_KEYWORDS) or any(
            kw in norm for kw in cls.WRITER_KEYWORDS
        ):
            return AgentMode.WRITER
        if re.search(
            r"\b(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?write\b", norm
        ):
            return AgentMode.WRITER
        if re.search(r"\bwrite\s+mode\b", norm):
            return AgentMode.WRITER
        if any(s in lower for s in cls.SWITCH_INDICATORS) and any(
            k in lower
            for k in ("writer", "writing", "\u4f5c\u8005", "\u6587\u6a94", "\u6587\u6863")
        ):
            return AgentMode.WRITER

        # 4. Coder checks
        if any(
            kw in lower
            for kw in (
                "implement directly",
                "skip planning",
                "code directly",
                "coder mode",
                "coding mode",
            )
        ):
            return AgentMode.CODER
        if re.search(
            r"\b(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?code\b", norm
        ):
            return AgentMode.CODER
        if re.search(r"\bcode\s+mode\b", norm):
            return AgentMode.CODER
        if any(s in lower for s in cls.SWITCH_INDICATORS) and any(
            k in lower
            for k in (
                "coder",
                "coding",
                "code",
                "\u7de8\u7a0b",
                "\u7f16\u7a0b",
                "\u5beb\u7a0b\u5f0f",
                "\u5199\u4ee3\u7801",
            )
        ):
            return AgentMode.CODER

        # 5. Chat checks
        if any(kw in lower for kw in ("switch to chat", "chat mode", "just chatting")):
            return AgentMode.CHAT
        if re.search(
            r"\b(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?chat\b", norm
        ):
            return AgentMode.CHAT
        if re.search(r"\bchat\s+mode\b", norm):
            return AgentMode.CHAT
        if any(s in lower for s in cls.SWITCH_INDICATORS) and any(
            k in lower for k in ("chat", "chatting", "\u804a\u5929", "\u5c0d\u8a71", "\u5bf9\u8bdd")
        ):
            return AgentMode.CHAT

        return None

    @classmethod
    def extract_clean_goal(cls, text: str) -> str:
        """
        Strip mode-switch directives from user input so downstream agents receive a clean task objective.
        E.g. 'ok can you change to plaaan mod and plaaan a small program...' -> 'plan a small program...'
        """
        stripped = text.strip()
        pattern = re.compile(
            r"^(?:(?:ok(?:ay)?|hey|hi|yo|please|can\s+you|could\s+you|would\s+you)[\s,;]*)*"
            r"(?:switch|change|turn|set|enter|go)\s+(?:to|into)?\s*(?:the\s*)?"
            r"(?:p+l+a+n+[a-z]*|c+o+d+[a-z]*|r+e+v+i+e+w+[a-z]*|w+r+i+t+[a-z]*|c+h+a+t+[a-z]*)\s*"
            r"(?:m+o+d+e*|\b)[\s,;]*(?:and|then|\&|&)?\s*",
            re.IGNORECASE,
        )
        cleaned = pattern.sub("", stripped).strip()
        return cleaned if cleaned else stripped

    @classmethod
    def classify(cls, text: str, current_mode: AgentMode = AgentMode.AUTO) -> AgentMode:
        """
        Classify user text into an AgentMode.
        Respects explicit switch requests even if not currently in AUTO mode.
        """
        stripped = text.strip()
        if not stripped:
            return AgentMode.CHAT

        # 1. Check if user explicitly asked to change mode in this prompt
        explicit = cls.detect_explicit_mode_request(stripped)
        if explicit:
            return explicit

        # 2. If current mode is not AUTO and no explicit override was detected, stay in current mode
        if current_mode != AgentMode.AUTO:
            return current_mode

        # 3. AUTO classification: Check greetings and direct conversational queries
        if cls.GREETINGS_PATTERN.match(stripped):
            return AgentMode.CHAT

        lower = stripped.lower()
        norm = cls.normalize_text(stripped)

        if any(q in lower for q in cls.CONVERSATIONAL_QUESTIONS):
            return AgentMode.CHAT

        # Conceptual questions without coding requests
        if any(
            lower.startswith(prefix)
            for prefix in (
                "what is",
                "why is",
                "how to",
                "explain",
                "describe",
                "\u4ec0\u9ebc\u662f",
                "\u4ec0\u4e48\u662f",
                "\u70ba\u4ec0\u9ebc",
                "\u4e3a\u4ec0\u4e48",
                "\u5982\u4f55",
                "\u89e3\u91cb",
            )
        ):
            if not any(k in lower for k in cls.CODE_ACTION_KEYWORDS):
                return AgentMode.CHAT

        # 4. Check for review requests
        if any(k in lower for k in cls.REVIEW_KEYWORDS) or any(
            k in norm for k in cls.REVIEW_KEYWORDS
        ):
            return AgentMode.REVIEWER

        # 5. Check for writer requests (evaluated before general code keywords)
        if any(k in lower for k in cls.WRITER_KEYWORDS) or any(
            k in norm for k in cls.WRITER_KEYWORDS
        ):
            return AgentMode.WRITER

        # 6. Check for planning requests
        if any(k in lower for k in cls.PLAN_KEYWORDS) or re.search(r"\bplan\b", norm):
            return AgentMode.PLANNER

        # In AUTO mode, multi-step creation or building of whole programs/apps routes to PLANNER
        if any(
            v in norm
            for v in ("create", "build", "develop", "implement", "make", "design", "write")
        ) and any(
            o in norm
            for o in ("program", "app", "application", "system", "tool", "project", "software")
        ):
            return AgentMode.PLANNER

        # 7. Check for coding requests
        if any(k in lower for k in cls.CODE_ACTION_KEYWORDS):
            if any(
                term in lower
                for term in (
                    "system",
                    "architecture",
                    "full project",
                    "complete app",
                    "\u67b6\u69cb",
                    "\u67b6\u6784",
                    "\u5b8c\u6574\u9805\u76ee",
                )
            ):
                return AgentMode.PLANNER
            return AgentMode.CODER

        # Default fallback for general open queries: CHAT mode is the safest and lowest overhead
        return AgentMode.CHAT

    @staticmethod
    def detect_plan_approval(user_input: str) -> str:
        """
        Classify user response to a proposed plan.
        Returns:
            - 'approve': User accepted the plan and wants to execute it
            - 'reject': User rejected/cancelled the plan
            - 'feedback': User provided revision guidance or other input
        """
        stripped = user_input.strip().lower()
        normalized = re.sub(r"[,.!?:;~/\s]+", " ", stripped).strip()

        approve_tokens = {
            "yes",
            "y",
            "approve",
            "approved",
            "ok",
            "okay",
            "proceed",
            "go",
            "sure",
            "sounds good",
            "start",
            "lgtm",
            "confirmed",
            "looks",
            "good",
            "great",
            "fine",
            "cool",
            "\u597d",
            "\u597d\u554a",
            "\u597d\u7684",
            "\u597d\u5440",
            "\u53ef\u4ee5",
            "\u53ef",
            "\u884c",
            "\u6c92\u554f\u984c",
            "\u6ca1\u95ee\u9898",
            "\u540c\u610f",
            "\u63a5\u53d7",
            "\u57f7\u884c",
            "\u6267\u884c",
            "\u958b\u59cb",
            "\u5f00\u59cb",
            "\u505a\u5427",
            "\u52d5\u624b",
            "\u8d70\u8d77",
            "\u597d\u6ef4",
            "\u6069",
            "\u6b63\u786e",
            "\u6c92\u932f",
            "\u6ca1\u9519",
        }

        approve_phrases = [
            "looks good",
            "sounds good",
            "go ahead",
            "let's do it",
            "lets do it",
            "\u6c92\u554f\u984c",
            "\u597d\u7684\u958b\u59cb",
            "\u597d\u554a\u958b\u59cb",
        ]
        for p in approve_phrases:
            if p in normalized:
                if any(
                    k in normalized
                    for k in (
                        "change",
                        "add",
                        "use",
                        "switch",
                        "replace",
                        "modify",
                        "update",
                        "except",
                        "but",
                    )
                ):
                    return "feedback"
                return "approve"

        if normalized in approve_tokens:
            return "approve"

        words = normalized.split()
        if words and all(w in approve_tokens for w in words):
            return "approve"

        # Check if individual Chinese approval character is present without negation
        if any(
            tok in stripped
            for tok in (
                "\u597d\u554a",
                "\u597d\u7684",
                "\u597d",
                "\u53ef\u4ee5",
                "\u6c92\u554f\u984c",
                "\u57f7\u884c",
            )
        ):
            if not any(
                neg in stripped for neg in ("\u4e0d", "\u5225", "\u53d6\u6d88", "\u4f46", "\u6539")
            ):
                return "approve"

        reject_tokens = {
            "no",
            "n",
            "reject",
            "rejected",
            "cancel",
            "stop",
            "nevermind",
            "abort",
            "quit",
            "disapprove",
            "\u4e0d\u8981",
            "\u4e0d\u884c",
            "\u53d6\u6d88",
            "\u505c\u6b62",
            "\u653e\u68c4",
            "\u653e\u5f03",
            "\u7b97\u4e86",
            "\u4e0d\u597d",
            "\u4e0d\u540c\u610f",
            "\u91cd\u4f86",
        }
        if normalized in reject_tokens:
            return "reject"
        if words and all(w in reject_tokens for w in words):
            return "reject"

        # Check if starts with approval but has modifiers/feedback
        if words and words[0] in ("yes", "ok", "okay") and len(words) > 1:
            remainder = " ".join(words[1:])
            if any(
                k in remainder
                for k in ("change", "add", "use", "switch", "replace", "modify", "update")
            ):
                return "feedback"
            if all(w in approve_tokens for w in words):
                return "approve"

        return "feedback"
