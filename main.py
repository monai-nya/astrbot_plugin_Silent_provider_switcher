from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Optional

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
    "YourName",
    "在 LLM 出错时静默切换到备用提供商。",
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

    def _parse_fallback_providers(self) -> List[Dict[str, str]]:
        raw = str(self.config.get("fallback_providers", "")).strip()
        if not raw:
            fallback_id = self._get_fallback_provider_id()
            if not fallback_id:
                return []
            return [{"provider_id": fallback_id}]
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = [data]
            items: List[Dict[str, str]] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    provider_id = str(item.get("provider_id", "")).strip()
                    if not provider_id:
                        continue
                    items.append(
                        {
                            "provider_id": provider_id,
                            "base_url": str(item.get("base_url", "")).strip(),
                            "api_key": str(item.get("api_key", "")).strip(),
                            "model": str(item.get("model", "")).strip(),
                        }
                    )
            return items
        except Exception:
            items = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                while len(parts) < 4:
                    parts.append("")
                provider_id, base_url, api_key, model = parts[:4]
                if provider_id:
                    items.append(
                        {
                            "provider_id": provider_id,
                            "base_url": base_url,
                            "api_key": api_key,
                            "model": model,
                        }
                    )
            return items

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

    def _apply_provider_overrides(
        self,
        provider: Any,
        base_url: str,
        api_key: str,
    ) -> List[tuple]:
        restores: List[tuple] = []
        if base_url:
            for attr in ("base_url", "api_base", "api_base_url", "endpoint", "host"):
                if hasattr(provider, attr):
                    old = getattr(provider, attr)
                    if old != base_url:
                        setattr(provider, attr, base_url)
                        restores.append(("attr", attr, old))
            cfg = getattr(provider, "config", None)
            if isinstance(cfg, dict):
                old = cfg.get("base_url")
                if old != base_url:
                    cfg["base_url"] = base_url
                    restores.append(("config", "base_url", old))
        if api_key:
            for attr in ("api_key", "key", "token", "access_token"):
                if hasattr(provider, attr):
                    old = getattr(provider, attr)
                    if old != api_key:
                        setattr(provider, attr, api_key)
                        restores.append(("attr", attr, old))
            cfg = getattr(provider, "config", None)
            if isinstance(cfg, dict):
                old = cfg.get("api_key")
                if old != api_key:
                    cfg["api_key"] = api_key
                    restores.append(("config", "api_key", old))
        return restores

    def _restore_provider_overrides(self, provider: Any, restores: List[tuple]) -> None:
        for kind, key, old in restores:
            if kind == "attr":
                if hasattr(provider, key):
                    setattr(provider, key, old)
            elif kind == "config":
                cfg = getattr(provider, "config", None)
                if isinstance(cfg, dict):
                    if old is None:
                        cfg.pop(key, None)
                    else:
                        cfg[key] = old

    async def _call_fallback(self, snapshot: _RequestSnapshot) -> Optional[LLMResponse]:
        fallbacks = self._parse_fallback_providers()
        if not fallbacks:
            return None
        for entry in fallbacks:
            provider_id = entry.get("provider_id", "")
            if snapshot.provider_id and snapshot.provider_id == provider_id:
                continue
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning("Fallback provider not found: %s", provider_id)
                continue
            kwargs: Dict[str, Any] = {
                "prompt": snapshot.prompt or "",
                "session_id": None,
                "contexts": snapshot.contexts or [],
                "image_urls": snapshot.image_urls or [],
                "func_tool": snapshot.func_tool_manager,
                "system_prompt": snapshot.system_prompt or "",
            }
            model = entry.get("model") or self._get_fallback_model() or snapshot.model
            if model:
                kwargs["model"] = model
            restores = self._apply_provider_overrides(
                provider,
                entry.get("base_url", ""),
                entry.get("api_key", ""),
            )
            try:
                try:
                    resp = await provider.text_chat(**kwargs)
                except TypeError:
                    if "model" in kwargs:
                        kwargs.pop("model", None)
                        resp = await provider.text_chat(**kwargs)
                    else:
                        raise
                if resp and not self._is_error_response(resp):
                    return resp
            except Exception:
                logger.exception("Fallback provider call failed.")
            finally:
                self._restore_provider_overrides(provider, restores)
        return None

    @filter.on_llm_request()
    async def capture_request(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._is_enabled():
            return
        if not self._parse_fallback_providers():
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
