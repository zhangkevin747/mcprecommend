import os
from typing import Dict, List, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    QWEN_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    KIMI_API_KEY: Optional[str] = None

    # Custom OpenAI-compatible API settings
    CUSTOM_OPENAI_API_KEY: Optional[str] = None
    CUSTOM_OPENAI_BASE_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",  
        env_file_encoding="utf-8",
        extra="ignore"            
    )

settings = Settings()

## Model Name Enum
# Claude
MODEL_SELECTION_CLAUDE_OPUS_4 = "claude-opus-4"
MODEL_SELECTION_CLAUDE_37 =  "claude-3-7-sonnet-20250219"
MODEL_SELECTION_CLAUDE_SONNET_46 = "claude-sonnet-4-6"
# OpenAI
MODEL_SELECTION_GPT4O =  "gpt-4o"
MODEL_SELECTION_GPT4O_MINI = "gpt-4o-mini"
MODEL_SELECTION_GPT5 = "gpt-5"
# Gemini
MODEL_SELECTION_GEMINI_25_FLASH = "gemini-2.5-flash"
# Qwen
MODEL_SELECTION_QWEN25_MAX = "qwen-max" # latest update to Qwen2.5
MODEL_SELECTION_QWEN3_PLUS = "qwen-plus"
MODEL_SELECTION_QWEN3_TURBO = "qwen-turbo"
MODEL_SELECTION_QWEN3_235B = "qwen3-235b-a22b-instruct-2507"
MODEL_SELECTION_QWEN3_CODER = "qwen3-coder-plus"
# Deepseek
MODEL_SELECTION_DEEPSEEK_R1 = "deepseek-r1"
# Kimi
MODEL_SELECTION_KIMI_K2 = "kimi-k2-0711-preview"

## Constant KEY 
KEY_MCP_TOOLS_DICT = "mcp_tools_dict"
KEY_BASE_COMPARE_FUNC = "base_compare_func"
KEY_COMPLETION = "completion"
KEY_REASON_CONTENT = "reason"
KEY_FUNCTION_CALL = "function_call"
