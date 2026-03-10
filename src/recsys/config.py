"""Configuration and constants for Experiment 2."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

# API keys
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def _load_smithery_key() -> str:
    """Load Smithery API key from CLI settings (smry_ token)."""
    settings_path = Path.home() / "Library" / "Application Support" / "smithery" / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text())
            return data.get("apiKey", "")
        except Exception:
            pass
    return os.environ.get("SMITHERY_API_KEY", "")


SMITHERY_API_KEY = _load_smithery_key()

# Data paths
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
POOL_DIR = DATA_DIR / "pool"
MCP_INDEX_PATH = INDEX_DIR / "mcp_server_index.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
EMBEDDING_INDEX_PATH = INDEX_DIR / "embedding_index.json"
SMITHERY_DETAILS_PATH = DATA_DIR / "raw" / "smithery_details.json"
MCP_POOL_PATH = POOL_DIR / "mcp_pool_300.json"
REPO_RESULTS_PATH = POOL_DIR / "repo_results_all.json"
TASKS_DIR = DATA_DIR  # benchmark tasks live in data/<domain>/

# Embedding config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# MCP server config
MCP_CONNECT_TIMEOUT = 30  # seconds to wait for server startup
MCP_CALL_TIMEOUT = 30     # seconds to wait for tool call

# Agent models
AGENTS = {
    "llama-4-maverick": {"provider": "openrouter", "model": "meta-llama/llama-4-maverick"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "gpt-5-mini": {"provider": "openai", "model": "gpt-5-mini"},
    "gemini-2.5-flash-lite": {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
    "grok-4-fast": {"provider": "openrouter", "model": "x-ai/grok-4-fast"},
    "deepseek-v3.2": {"provider": "openrouter", "model": "deepseek/deepseek-v3.2"},
}

# Default experiment params
DEFAULT_K = 5
DEFAULT_RETRIEVE_N = 100
