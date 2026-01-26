from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, register


@dataclass
class _RequestSnapshot:
    prompt: str
    contexts: list
    system_prompt: str
    image_urls: list
    func_tool_manager: Any
    model: Optional[str]
    provider_id: Optional[str]


@register(
    "silent_provider_switcher",
    "洛曦",
    "Silently retries failed LLM calls with a backup provider.",
    "1.0.0",
)
class SilentProviderSwitcher(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._req_cache: Dict[int, _RequestSnapshot] = {}

    def _is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _get_fallback_provider_id(self) -> str:
        return str(self.config.get("fallback_provider_id", "")).strip()

    def _get_fallback_model(self) -> Optional[str]:
        model = str(self.config.get("fallback_model", "")).strip()
        return model or None

    def _get_error_keywords(self) -> Iterable[str]:
        raw = str(self.config.get("error_keywords", "")).strip()
        if not raw:
            return []
        items = []
        for line in raw.replace(",", "\n").splitlines():
            line = line.strip()
            if line:
                items.append(line)
        return items

    def _should_log_switch(self) -> bool:
        return bool(self.config.get("log_switch", True))

    def _make_key(self, event: AstrMessageEvent) -> int:
        return id(event)

    def _is_error_response(self, resp: LLMResponse) -> bool:
        role = getattr(resp, "role", None)
        if role in {"err", "error"}:
            return True
        if getattr(resp, "error", None) is not None:
            return True
        if getattr(resp, "err", None) is not None:
            return True
        return False

    def _matches_error_keywords(self, resp: LLMResponse) -> bool:
        keywords = list(self._get_error_keywords())
        if not keywords:
            return False
        parts = [
            str(getattr(resp, "completion_text", "") or ""),
            str(getattr(resp, "error", "") or ""),
            str(getattr(resp, "err", "") or ""),
        ]
        haystack = " ".join(parts).lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def _apply_response(self, target: LLMResponse, source: LLMResponse) -> None:
        for attr in (
            "role",
            "completion_text",
            "result_chain",
            "tools_call_name",
            "tools_call_args",
            "tools_call_ids",
            "raw_completion",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    async def _call_fallback(self, snapshot: _RequestSnapshot) -> Optional[LLMResponse]:
        fallback_provider_id = self._get_fallback_provider_id()
        if not fallback_provider_id:
            return None
        if snapshot.provider_id and snapshot.provider_id == fallback_provider_id:
            return None
        provider = self.context.get_provider_by_id(fallback_provider_id)
        if not provider:
            logger.warning("Fallback provider not found: %s", fallback_provider_id)
            return None
        kwargs: Dict[str, Any] = {
            "prompt": snapshot.prompt or "",
            "session_id": None,
            "contexts": snapshot.contexts or [],
            "image_urls": snapshot.image_urls or [],
            "func_tool": snapshot.func_tool_manager,
            "system_prompt": snapshot.system_prompt or "",
        }
        model = self._get_fallback_model() or snapshot.model
        if model:
            kwargs["model"] = model
        try:
            return await provider.text_chat(**kwargs)
        except TypeError:
            if "model" in kwargs:
                kwargs.pop("model", None)
                try:
                    return await provider.text_chat(**kwargs)
                except Exception:
                    logger.exception("Fallback provider call failed.")
                    return None
            logger.exception("Fallback provider call failed.")
            return None
        except Exception:
            logger.exception("Fallback provider call failed.")
            return None

    @filter.on_llm_request()
    async def capture_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._is_enabled():
            return
        if not self._get_fallback_provider_id():
            return
        key = self._make_key(event)
        snapshot = _RequestSnapshot(
            prompt=str(getattr(req, "prompt", "") or ""),
            contexts=list(getattr(req, "contexts", []) or []),
            system_prompt=str(getattr(req, "system_prompt", "") or ""),
            image_urls=list(getattr(req, "image_urls", []) or []),
            func_tool_manager=getattr(req, "func_tool_manager", None)
            or getattr(req, "func_tool", None),
            model=getattr(req, "model", None),
            provider_id=getattr(req, "provider_id", None)
            or getattr(req, "chat_provider_id", None),
        )
        self._req_cache[key] = snapshot

    @filter.on_llm_response()
    async def handle_response(self, event: AstrMessageEvent, resp: LLMResponse):
        if not self._is_enabled():
            return
        key = self._make_key(event)
        snapshot = self._req_cache.pop(key, None)
        if not snapshot:
            return
        if not (self._is_error_response(resp) or self._matches_error_keywords(resp)):
            return
        fallback_resp = await self._call_fallback(snapshot)
        if not fallback_resp:
            return
        if self._is_error_response(fallback_resp):
            return
        if self._should_log_switch():
            logger.info(
                "Switched to fallback provider for event %s.",
                getattr(event, "unified_msg_origin", "unknown"),
            )
        self._apply_response(resp, fallback_resp)
