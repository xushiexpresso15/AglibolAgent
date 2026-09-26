"""Asynchronous HTTP client for interacting with the local Ollama daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import httpx

from aglibol.core.types import ChatMessage, ToolCall
from aglibol.ollama.models import LoadedModel, ModelDetail, ModelInfo

logger = logging.getLogger("aglibol.ollama")


class OllamaClient:
    """High-performance async client for the Ollama REST API with retry logic and VRAM control."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 180,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=self.timeout)
                res = await self._client.request(method, url, **kwargs)
                res.raise_for_status()
                return res
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    self._client = httpx.AsyncClient(timeout=self.timeout)
                    res = await self._client.request(method, url, **kwargs)
                    res.raise_for_status()
                    return res
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Ollama request failed: {e}. Retrying in {delay}s (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Ollama request failed after {self.max_retries} retries: {e}")
                    raise
            except httpx.HTTPStatusError:
                # Do not retry on HTTP status errors (like 404, 400)
                raise
        raise last_exception  # type: ignore

    async def health_check(self) -> bool:
        """Check if Ollama service is reachable and running."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as check_client:
                res = await check_client.get(f"{self.host}/api/version")
                return res.status_code == 200
        except Exception:
            return False

    async def start_daemon(self, timeout_seconds: float = 10.0) -> bool:
        """
        Attempt to start the local Ollama daemon if installed.
        Returns True if the daemon becomes healthy within the timeout window.
        """
        if await self.health_check():
            return True

        ollama_bin = shutil.which("ollama")
        app_exe: Path | None = None
        if sys.platform == "win32":
            candidate = (
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
            )
            if candidate.exists():
                app_exe = candidate

        if not ollama_bin and not app_exe:
            logger.warning(
                "Ollama executable not found in PATH or standard installation directory."
            )
            return False

        try:
            if sys.platform == "win32":
                cmd = [str(app_exe)] if app_exe else ["ollama", "serve"]
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
                subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            # Poll health check
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            while loop.time() - start_time < timeout_seconds:
                await asyncio.sleep(0.5)
                if await self.health_check():
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to start Ollama daemon: {e}")
            return False

    async def list_models(self) -> list[ModelInfo]:
        """Fetch all installed models from /api/tags."""
        url = f"{self.host}/api/tags"
        try:
            res = await self._request_with_retry("GET", url)
            data = res.json()
            models: list[ModelInfo] = []
            for m in data.get("models", []):
                details = m.get("details", {})
                models.append(
                    ModelInfo(
                        name=m.get("name", ""),
                        size_bytes=m.get("size", 0),
                        digest=m.get("digest", ""),
                        modified_at=m.get("modified_at", ""),
                        parameter_size=details.get("parameter_size", ""),
                        quantization_level=details.get("quantization_level", ""),
                    )
                )
            return models
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    async def ps(self) -> list[LoadedModel]:
        """List currently loaded models in RAM/VRAM from /api/ps."""
        url = f"{self.host}/api/ps"
        try:
            res = await self._request_with_retry("GET", url)
            data = res.json()
            loaded: list[LoadedModel] = []
            for m in data.get("models", []):
                loaded.append(
                    LoadedModel(
                        name=m.get("name", ""),
                        model=m.get("model", ""),
                        size=m.get("size", 0),
                        digest=m.get("digest", ""),
                        expires_at=m.get("expires_at", ""),
                        size_vram=m.get("size_vram", 0),
                    )
                )
            return loaded
        except Exception as e:
            logger.warning(f"Failed to get loaded models: {e}")
            return []

    async def unload_model(self, model_name: str) -> bool:
        """Immediately evict a model from VRAM by setting keep_alive to 0."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": model_name,
            "keep_alive": 0,
        }
        try:
            res = await self._request_with_retry("POST", url, json=payload, timeout=10.0)
            return res.status_code == 200
        except Exception:
            return False

    async def preload_model(self, model: str) -> None:
        """Warm-up / preload a model into VRAM."""
        url = f"{self.host}/api/generate"
        try:
            await self._client.post(url, json={"model": model, "keep_alive": -1}, timeout=5.0)
        except Exception:
            pass

    async def show_model(self, model_name: str) -> ModelDetail | None:
        """Get model parameters, system prompt, and template via /api/show."""
        url = f"{self.host}/api/show"
        try:
            res = await self._request_with_retry("POST", url, json={"name": model_name})
            data = res.json()
            return ModelDetail(
                license=data.get("license", ""),
                modelfile=data.get("modelfile", ""),
                parameters=data.get("parameters", ""),
                template=data.get("template", ""),
                system=data.get("system", ""),
                details=data.get("details", {}),
            )
        except Exception:
            return None

    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        tools: list[dict[str, Any]] | None = None,
        format: dict | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat completion tokens from /api/chat."""
        url = f"{self.host}/api/chat"

        # Format messages for Ollama API
        formatted_messages: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {"function": {"name": tc.tool_name, "arguments": tc.arguments}}
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(m)

        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format

        async with self._client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                raw = await response.aread()
                err_text = raw.decode(errors="replace")
                try:
                    err_json = json.loads(err_text)
                    err_msg = err_json.get("error", err_text)
                except Exception:
                    err_msg = err_text

                if response.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{model}' not found. Please install it using 'ollama pull {model}' or select an installed model via '/model'."
                    )
                raise RuntimeError(f"Ollama error ({response.status_code}): {err_msg}")

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        tools: list[dict[str, Any]] | None = None,
        format: dict | None = None,
        on_token: Callable[[str, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a chat request, streaming tokens internally and accumulating response and thoughts."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final_chunk: dict[str, Any] = {}

        in_think_tag = False
        async for chunk in self.chat_stream(
            model=model,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
            tools=tools,
            format=format,
        ):
            final_chunk = chunk
            msg = chunk.get("message", {})
            if "thinking" in msg and msg["thinking"]:
                thinking_text = msg["thinking"]
                thinking_parts.append(thinking_text)
                if on_token:
                    try:
                        res = on_token("thinking", thinking_text)
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:
                        pass
            if "content" in msg and msg["content"]:
                content_text = msg["content"]
                if "<think>" in content_text or "<thinking>" in content_text:
                    in_think_tag = True

                if in_think_tag:
                    cleaned_think = re.sub(r"<(?:think|thinking)>", "", content_text)
                    if "</think>" in cleaned_think or "</thinking>" in cleaned_think:
                        in_think_tag = False
                        cleaned_think = re.sub(r"</(?:think|thinking)>", "", cleaned_think)
                    if cleaned_think:
                        thinking_parts.append(cleaned_think)
                        if on_token:
                            try:
                                res = on_token("thinking", cleaned_think)
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                pass
                else:
                    content_parts.append(content_text)
                    if on_token:
                        try:
                            res = on_token("content", content_text)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception:
                            pass
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append(
                        ToolCall(
                            tool_name=fn.get("name", ""),
                            arguments=args,
                        )
                    )

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)

        # Fallback: Extract thinking from XML think tags if not emitted separately
        if not full_thinking and ("<think>" in full_content or "<thinking>" in full_content):
            m = re.search(r"<(?:think|thinking)>([\s\S]*?)</(?:think|thinking)>", full_content)
            if m:
                full_thinking = m.group(1).strip()
                full_content = re.sub(
                    r"<(?:think|thinking)>[\s\S]*?</(?:think|thinking)>", "", full_content
                ).strip()

        # Fallback: Parse XML <tool_call> or markdown tool_call blocks if model output text instead of API tool_calls
        if not tool_calls and (full_content or full_thinking) and tools:
            tool_calls = self._parse_fallback_tool_calls(full_content or full_thinking)

        return {
            "content": full_content,
            "thinking": full_thinking,
            "tool_calls": tool_calls,
            "done": final_chunk.get("done", True),
            "done_reason": final_chunk.get("done_reason", "stop"),
            "eval_count": final_chunk.get("eval_count", 0),
            "eval_duration": final_chunk.get("eval_duration", 0),
            "prompt_eval_count": final_chunk.get("prompt_eval_count", 0),
        }

    @staticmethod
    def _parse_fallback_tool_calls(text: str) -> list[ToolCall]:
        """Extract tool calls from model text output formatted as <tool_call> or ```tool_call."""
        extracted: list[ToolCall] = []

        xml_matches = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
        md_matches = re.findall(r"```tool_call\s*([\s\S]*?)\s*```", text, re.IGNORECASE)

        for block in xml_matches + md_matches:
            cleaned = block.strip()
            try:
                data = json.loads(cleaned)
            except Exception:
                # Heuristic repair for single quotes
                repaired = re.sub(r"'([a-zA-Z0-9_\-]+)'\s*:", r'"\1":', cleaned)
                repaired = re.sub(
                    r"(?<=[{\[,:])\s*'([^'\\]*(?:\\.[^'\\]*)*)'(?=\s*[:,\]}])", r'"\1"', repaired
                )
                try:
                    data = json.loads(repaired)
                except Exception:
                    continue

            if isinstance(data, dict):
                name = (
                    data.get("name") or data.get("tool") or (data.get("function") or {}).get("name")
                )
                args = (
                    data.get("arguments")
                    or data.get("args")
                    or data.get("parameters")
                    or (data.get("function") or {}).get("arguments", {})
                )
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    extracted.append(
                        ToolCall(
                            tool_name=str(name),
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )

        return extracted
