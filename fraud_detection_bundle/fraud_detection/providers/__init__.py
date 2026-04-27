"""LLM provider abstraction.

The forensic detectors run independently of any LLM (they are deterministic
Python). A *provider* is the optional reasoning layer on top:

  * AnthropicProvider — Claude with vision; reads images directly to
    corroborate detector findings.
  * DeepSeekProvider  — DeepSeek's OpenAI-compatible chat API; reasons
    over the detector JSON only (no vision).
  * OfflineProvider   — deterministic local fusion, no API calls.

Adding another provider is one file: implement ``Provider.summarize`` and
register it in ``build_provider``.
"""

from fraud_detection.providers.anthropic_provider import AnthropicProvider
from fraud_detection.providers.base import (
    InputManifest,
    Provider,
    ProviderInput,
    ProviderResult,
)
from fraud_detection.providers.deepseek_provider import DeepSeekProvider
from fraud_detection.providers.factory import build_provider, list_providers
from fraud_detection.providers.offline_provider import OfflineProvider
from fraud_detection.providers.openai_provider import OpenAIProvider

__all__ = [
    "InputManifest",
    "Provider",
    "ProviderInput",
    "ProviderResult",
    "AnthropicProvider",
    "DeepSeekProvider",
    "OfflineProvider",
    "OpenAIProvider",
    "build_provider",
    "list_providers",
]
