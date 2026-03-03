from typing import Dict, Any

from ..global_variables import *
from .qwen_api import QwenModelAPIProvider
from .kimi_api import KimiModelAPIProvider
from .claude_api import ClaudeModelAPIProvider
from .openai_api import OpenAIModelAPIProvider
from .custom_openai_api import CustomOpenAIAPIProvider

_global_model_provider: Dict[str, Any] = {}

## CLAUDE
if settings.ANTHROPIC_API_KEY:
    _global_model_provider[MODEL_SELECTION_CLAUDE_37] = ClaudeModelAPIProvider(MODEL_SELECTION_CLAUDE_37)
    _global_model_provider[MODEL_SELECTION_CLAUDE_OPUS_4] = ClaudeModelAPIProvider(MODEL_SELECTION_CLAUDE_OPUS_4)
    _global_model_provider[MODEL_SELECTION_CLAUDE_SONNET_46] = ClaudeModelAPIProvider(MODEL_SELECTION_CLAUDE_SONNET_46)

## OPENAI
if settings.OPENAI_API_KEY:
    _global_model_provider[MODEL_SELECTION_GPT4O] = OpenAIModelAPIProvider(MODEL_SELECTION_GPT4O)
    _global_model_provider[MODEL_SELECTION_GPT4O_MINI] = OpenAIModelAPIProvider(MODEL_SELECTION_GPT4O_MINI)
    _global_model_provider[MODEL_SELECTION_GPT5] = OpenAIModelAPIProvider(MODEL_SELECTION_GPT5)

## QWEN
if settings.QWEN_API_KEY:
    _global_model_provider[MODEL_SELECTION_QWEN25_MAX] = QwenModelAPIProvider(MODEL_SELECTION_QWEN25_MAX)
    _global_model_provider[MODEL_SELECTION_QWEN3_PLUS] = QwenModelAPIProvider(MODEL_SELECTION_QWEN3_PLUS)
    _global_model_provider[MODEL_SELECTION_QWEN3_TURBO] = QwenModelAPIProvider(MODEL_SELECTION_QWEN3_TURBO)
    _global_model_provider[MODEL_SELECTION_QWEN3_235B] = QwenModelAPIProvider(MODEL_SELECTION_QWEN3_235B)
    _global_model_provider[MODEL_SELECTION_QWEN3_CODER] = QwenModelAPIProvider(MODEL_SELECTION_QWEN3_CODER)

## KIMI
if settings.KIMI_API_KEY:
    _global_model_provider[MODEL_SELECTION_KIMI_K2] = KimiModelAPIProvider(MODEL_SELECTION_KIMI_K2)


def get_model_provider(model: str):
    """
    Get or create a model provider for the given model.
    If the model exists in _global_model_provider, return it.
    Otherwise, try to create a CustomOpenAI provider if custom settings are available.

    Args:
        model: The model name to get/create provider for

    Returns:
        Model provider instance or None if not available
    """
    # Check if model already exists in global provider
    if model in _global_model_provider:
        return _global_model_provider[model]

    # If custom OpenAI settings are available, create a dynamic provider
    if settings.CUSTOM_OPENAI_BASE_URL and settings.CUSTOM_OPENAI_API_KEY:
        # Create a new CustomOpenAI provider with the requested model name
        provider = CustomOpenAIAPIProvider(
            model_name=model,
            base_url=settings.CUSTOM_OPENAI_BASE_URL,
            api_key=settings.CUSTOM_OPENAI_API_KEY
        )
        # Cache it for future use
        _global_model_provider[model] = provider
        return provider

    return None
