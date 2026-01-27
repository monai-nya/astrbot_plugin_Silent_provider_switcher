from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from astrbot.api import AstrBotConfig, logger as astr_logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.provider.entities import ProviderType, LLMResponse
except Exception:
    from astrbot.api.provider import LLMResponse
    ProviderType = None


@dataclass
class _FallbackEntry:
    provider_id: str
    base_url: str = ""
    api_key: str = ""
    model: str = ""


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
        self._wrapped: set[int] = set()

    async def initialize(self):
        self._install_provider_failover()

    @filter.on_astrbot_loaded()
    async def on_bot_loaded(self):
        self._install_provider_failover()

    def _is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _should_log_switch(self) -> bool:
        return bool(self.config.get("log_switch", True))

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

    def _get_fallback_entries(self) -> List[_FallbackEntry]:
        entries: List[_FallbackEntry] = []
        for idx in range(1, 4):
            provider_id = str(
                self.config.get(f"fallback_provider_id_{idx}", "")
            ).strip()
            if not provider_id:
                continue
            entries.append(
                _FallbackEntry(
                    provider_id=provider_id,
                    base_url=str(
                        self.config.get(f"fallback_base_url_{idx}", "")
                    ).strip(),
                    api_key=str(
                        self.config.get(f"fallback_api_key_{idx}", "")
                    ).strip(),
                    model=str(
                        self.config.get(f"fallback_model_{idx}", "")
                    ).strip(),
                )
            )
        if not entries:
            fallback_id = str(self.config.get("fallback_provider_id", "")).strip()
            if fallback_id:
                entries.append(
                    _FallbackEntry(
                        provider_id=fallback_id,
                        model=str(self.config.get("fallback_model", "")).strip(),
                    )
                )
        return entries

    def _get_provider_id(self, provider: Any) -> str:
        if not provider:
            return "unknown"
        if hasattr(provider, "provider_config"):
            try:
                value = provider.provider_config.get("id")
                if value:
                    return str(value)
            except Exception:
                pass
        for attr in ("provider_id", "id", "name"):
            if hasattr(provider, attr):
                try:
                    value = getattr(provider, attr)
                    if value:
                        return str(value)
                except Exception:
                    continue
        return "unknown"

    def _get_all_chat_providers(self) -> List[Any]:
        providers: List[Any] = []
        if hasattr(self.context, "get_all_providers"):
            try:
                providers = list(self.context.get_all_providers())
            except Exception:
                providers = []
        if not providers:
            for attr in (
                "providers",
                "_providers",
                "provider_manager",
                "provider_pool",
                "provider_registry",
                "llm_providers",
            ):
                if hasattr(self.context, attr):
                    obj = getattr(self.context, attr)
                    providers = self._flatten_providers(obj)
                    if providers:
                        break
        if ProviderType is None:
            return providers
        filtered = []
        for provider in providers:
            try:
                if provider.meta().provider_type == ProviderType.CHAT_COMPLETION:
                    filtered.append(provider)
            except Exception:
                filtered.append(provider)
        return filtered

    def _flatten_providers(self, obj: Any) -> List[Any]:
        if not obj:
            return []
        if isinstance(obj, dict):
            return [v for v in obj.values() if v]
        if isinstance(obj, (list, tuple, set)):
            return [v for v in obj if v]
        for attr in ("providers", "_providers", "provider_map", "provider_dict"):
            if hasattr(obj, attr):
                try:
                    return self._flatten_providers(getattr(obj, attr))
                except Exception:
                    pass
        for method in ("get_all_providers", "list_providers", "get_providers"):
            if hasattr(obj, method):
                try:
                    return self._flatten_providers(getattr(obj, method)())
                except Exception:
                    pass
        return []

    def _should_failover_exception(self, exc: Exception) -> bool:
        code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(code, int) and code in {
            401,
            402,
            403,
            408,
            409,
            429,
            500,
            502,
            503,
            504,
        }:
            return True
        message = str(exc).lower()
        keywords = [k.lower() for k in self._get_error_keywords()]
        default_keywords = [
            "rate limit",
            "too many requests",
            "timeout",
            "timed out",
            "connection reset",
            "invalid api key",
            "authentication error",
        ]
        if not keywords:
            keywords = default_keywords
        else:
            keywords = keywords + default_keywords
        return any(keyword in message for keyword in keywords)

    def _is_error_response(self, resp: Any) -> bool:
        if not resp:
            return False
        if isinstance(resp, LLMResponse):
            role = getattr(resp, "role", None)
            if role in {"err", "error"}:
                return True
            if getattr(resp, "error", None) is not None:
                return True
            if getattr(resp, "err", None) is not None:
                return True
        return False

    def _matches_error_keywords(self, resp: Any) -> bool:
        keywords = list(self._get_error_keywords())
        if not keywords or not resp:
            return False
        parts = [
            str(getattr(resp, "completion_text", "") or ""),
            str(getattr(resp, "error", "") or ""),
            str(getattr(resp, "err", "") or ""),
        ]
        haystack = " ".join(parts).lower()
        return any(keyword.lower() in haystack for keyword in keywords)

    def _apply_provider_overrides(
        self, provider: Any, base_url: str, api_key: str
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

    def _build_failover_plan(self, primary: Any):
        plan = [(primary, None, True)] if primary else []
        entries = self._get_fallback_entries()
        if not entries:
            return plan

        providers = {self._get_provider_id(p): p for p in self._get_all_chat_providers()}
        for entry in entries:
            provider = providers.get(entry.provider_id)
            if provider is None and hasattr(self.context, "get_provider_by_id"):
                try:
                    provider = self.context.get_provider_by_id(entry.provider_id)
                except Exception:
                    provider = None
            if provider is None:
                astr_logger.warning(
                    "Fallback provider not found: %s", entry.provider_id
                )
                continue
            if provider is primary and not (entry.base_url or entry.api_key or entry.model):
                continue
            plan.append((provider, entry, False))
        return plan

    def _install_provider_failover(self):
        if not self._is_enabled():
            return
        providers = self._get_all_chat_providers()
        if not providers:
            return
        for provider in providers:
            key = id(provider)
            if key in self._wrapped:
                continue
            if not hasattr(provider, "text_chat"):
                continue
            if not hasattr(provider, "_silent_provider_switcher_original_text_chat"):
                provider._silent_provider_switcher_original_text_chat = provider.text_chat
            if not getattr(provider, "_silent_provider_switcher_text_wrapped", False):
                async def wrapper(p_self, *args, **kwargs):
                    return await self._execute_with_failover(p_self, *args, **kwargs)
                provider.text_chat = types.MethodType(wrapper, provider)
                provider._silent_provider_switcher_text_wrapped = True

            if hasattr(provider, "text_chat_stream"):
                if not hasattr(
                    provider, "_silent_provider_switcher_original_text_chat_stream"
                ):
                    provider._silent_provider_switcher_original_text_chat_stream = (
                        provider.text_chat_stream
                    )
                if not getattr(provider, "_silent_provider_switcher_stream_wrapped", False):
                    async def stream_wrapper(p_self, *args, **kwargs):
                        async for chunk in self._execute_stream_with_failover(
                            p_self, *args, **kwargs
                        ):
                            yield chunk
                    provider.text_chat_stream = types.MethodType(
                        stream_wrapper, provider
                    )
                    provider._silent_provider_switcher_stream_wrapped = True

            self._wrapped.add(key)

    def _get_prompt_preview(self, args: tuple, kwargs: dict) -> str:
        if args:
            return str(args[0])[:80]
        if "prompt" in kwargs:
            return str(kwargs["prompt"])[:80]
        return ""

    async def _execute_with_failover(self, primary, *args, **kwargs):
        plan = self._build_failover_plan(primary)
        errors = []
        prompt_preview = self._get_prompt_preview(args, kwargs)

        for index, (provider, entry, is_primary) in enumerate(plan):
            provider_id = self._get_provider_id(provider)
            original_call = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat",
                provider.text_chat,
            )
            call_kwargs = dict(kwargs)
            if entry and entry.model:
                call_kwargs["model"] = entry.model
            restores = []
            if entry:
                restores = self._apply_provider_overrides(
                    provider, entry.base_url, entry.api_key
                )
            try:
                result = await original_call(*args, **call_kwargs)
                if self._is_error_response(result) or self._matches_error_keywords(result):
                    if index < len(plan) - 1:
                        errors.append((provider_id, RuntimeError("LLM error response")))
                        continue
                if not is_primary and self._should_log_switch():
                    astr_logger.info(
                        "Switched to fallback provider: %s (prompt=%s)",
                        provider_id,
                        prompt_preview or "[empty]",
                    )
                return result
            except Exception as exc:
                if self._should_failover_exception(exc) and index < len(plan) - 1:
                    errors.append((provider_id, exc))
                    continue
                raise
            finally:
                self._restore_provider_overrides(provider, restores)

        if errors:
            raise errors[-1][1]
        raise RuntimeError("No available provider")

    async def _execute_stream_with_failover(
        self, primary, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        plan = self._build_failover_plan(primary)
        errors = []
        prompt_preview = self._get_prompt_preview(args, kwargs)

        for index, (provider, entry, is_primary) in enumerate(plan):
            provider_id = self._get_provider_id(provider)
            original_stream = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat_stream",
                getattr(provider, "text_chat_stream", None),
            )
            original_call = getattr(
                provider,
                "_silent_provider_switcher_original_text_chat",
                getattr(provider, "text_chat", None),
            )

            if not original_stream and not original_call:
                errors.append((provider_id, RuntimeError("Missing chat method")))
                continue

            call_kwargs = dict(kwargs)
            if entry and entry.model:
                call_kwargs["model"] = entry.model
            restores = []
            if entry:
                restores = self._apply_provider_overrides(
                    provider, entry.base_url, entry.api_key
                )

            emitted = False
            try:
                if original_stream:
                    async for chunk in original_stream(*args, **call_kwargs):
                        emitted = True
                        yield chunk
                else:
                    result = await original_call(*args, **call_kwargs)
                    emitted = True
                    yield result

                if not is_primary and self._should_log_switch():
                    astr_logger.info(
                        "Switched to fallback provider(stream): %s (prompt=%s)",
                        provider_id,
                        prompt_preview or "[empty]",
                    )
                return
            except Exception as exc:
                if emitted:
                    raise
                if self._should_failover_exception(exc) and index < len(plan) - 1:
                    errors.append((provider_id, exc))
                    continue
                raise
            finally:
                self._restore_provider_overrides(provider, restores)

        if errors:
            raise errors[-1][1]
        raise RuntimeError("No available provider")
