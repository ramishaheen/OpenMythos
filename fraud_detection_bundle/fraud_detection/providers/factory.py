"""Provider factory + listing."""

from __future__ import annotations

from typing import Any

from fraud_detection.providers.anthropic_provider import (
    AnthropicProvider,
    DEFAULT_MODEL as DEFAULT_ANTHROPIC_MODEL,
)
from fraud_detection.providers.base import Provider
from fraud_detection.providers.deepseek_provider import (
    DEFAULT_MODEL as DEFAULT_DEEPSEEK_MODEL,
    DeepSeekProvider,
)
from fraud_detection.providers.offline_provider import OfflineProvider
from fraud_detection.providers.openai_provider import (
    DEFAULT_MODEL as DEFAULT_OPENAI_MODEL,
    OpenAIProvider,
)

PROVIDER_DEFAULTS = {
    "anthropic": {
        "default_model": DEFAULT_ANTHROPIC_MODEL,
        "models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"],
        "supports_vision": True,
    },
    "openai": {
        "default_model": DEFAULT_OPENAI_MODEL,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "supports_vision": True,
    },
    "deepseek": {
        "default_model": DEFAULT_DEEPSEEK_MODEL,
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "supports_vision": False,
    },
    "offline": {
        "default_model": "local-fusion",
        "models": ["local-fusion"],
        "supports_vision": False,
    },
}


def list_providers() -> dict[str, dict[str, Any]]:
    return PROVIDER_DEFAULTS


def build_provider(
    name: str | None,
    api_key: str | None = None,
    model: str | None = None,
) -> Provider:
    """Build a provider by name. Falls back to OfflineProvider if not configured."""
    name = (name or "offline").lower()
    if name == "anthropic":
        m = model or DEFAULT_ANTHROPIC_MODEL
        if m not in PROVIDER_DEFAULTS["anthropic"]["models"]:
            raise ValueError(f"unknown anthropic model: {m}")
        prov = AnthropicProvider(api_key=api_key, model=m)
        if not prov.configured():
            return OfflineProvider()
        return prov
    if name == "openai":
        m = model or DEFAULT_OPENAI_MODEL
        if m not in PROVIDER_DEFAULTS["openai"]["models"]:
            raise ValueError(f"unknown openai model: {m}")
        prov = OpenAIProvider(api_key=api_key, model=m)
        if not prov.configured():
            return OfflineProvider()
        return prov
    if name == "deepseek":
        m = model or DEFAULT_DEEPSEEK_MODEL
        if m not in PROVIDER_DEFAULTS["deepseek"]["models"]:
            raise ValueError(f"unknown deepseek model: {m}")
        prov = DeepSeekProvider(api_key=api_key, model=m)
        if not prov.configured():
            return OfflineProvider()
        return prov
    if name == "offline":
        return OfflineProvider()
    raise ValueError(f"unknown provider: {name}")
