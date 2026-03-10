"""Agent API wrapper: call Anthropic/OpenAI with MCP tool schemas."""

import json
import logging
import time
from typing import Awaitable, Callable

import anthropic
import openai

from .config import AGENTS, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
from .mcp_client import MCPServerConnection

LazyConnectFn = Callable[[str], Awaitable[object | None]]

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the available tools. "
    "After using a tool, give a short, direct answer based on the result."
)


def _sanitize(name: str) -> str:
    """Make a string safe for API tool names: [a-zA-Z0-9_-] only."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


DESC_MAX_LEN = 150  # truncate tool descriptions for first turn


def _compact_schema(schema: dict) -> dict:
    """Strip property descriptions from schema to save tokens."""
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    result = {}
    for k, v in schema.items():
        if k == "properties" and isinstance(v, dict):
            result["properties"] = {}
            for prop_name, prop_val in v.items():
                if isinstance(prop_val, dict):
                    # Keep type, enum, default, required — strip description
                    compact = {pk: pv for pk, pv in prop_val.items() if pk != "description"}
                    result["properties"][prop_name] = compact
                else:
                    result["properties"][prop_name] = prop_val
        else:
            result[k] = v
    return result


class ToolRegistry:
    """Maps API-safe tool names to (server_id, tool_name) pairs.

    Builds compact tool lists (truncated descriptions, stripped schemas) for
    the initial turn, and stores full schemas for retry turns.
    """

    def __init__(self):
        self._api_to_mcp: dict[str, tuple[str, str]] = {}
        self._compact_anthropic: list[dict] = []
        self._compact_openai: list[dict] = []
        self._full_anthropic: list[dict] = []
        self._full_openai: list[dict] = []

    def register(self, tools: list[dict]):
        """Register MCP tools and build compact + full API tool lists."""
        self._api_to_mcp.clear()
        self._compact_anthropic.clear()
        self._compact_openai.clear()
        self._full_anthropic.clear()
        self._full_openai.clear()

        seen: dict[str, int] = {}
        for t in tools:
            server_id = t["server_id"]
            tool_name = t["name"]
            base = f"s_{_sanitize(server_id)}__t_{_sanitize(tool_name)}"
            # OpenAI enforces max 64 char tool names
            if len(base) > 64:
                base = base[:64]
            if base in seen:
                seen[base] += 1
                api_name = f"{base[:60]}_{seen[base]}"
            else:
                seen[base] = 0
                api_name = base

            self._api_to_mcp[api_name] = (server_id, tool_name)

            full_desc = f"[{server_id}] {t.get('description', '')}"
            short_desc = full_desc[:DESC_MAX_LEN]

            schema = t.get("input_schema", {})
            if not schema:
                schema = {"type": "object", "properties": {}}
            compact = _compact_schema(schema)

            # Compact versions (for first turn)
            self._compact_anthropic.append({
                "name": api_name,
                "description": short_desc,
                "input_schema": compact,
            })
            self._compact_openai.append({
                "type": "function",
                "function": {
                    "name": api_name,
                    "description": short_desc,
                    "parameters": compact,
                },
            })

            # Full versions (for retry turns)
            self._full_anthropic.append({
                "name": api_name,
                "description": full_desc,
                "input_schema": schema,
            })
            self._full_openai.append({
                "type": "function",
                "function": {
                    "name": api_name,
                    "description": full_desc,
                    "parameters": schema,
                },
            })

    def resolve(self, api_name: str) -> tuple[str, str]:
        """Resolve API tool name → (server_id, tool_name)."""
        return self._api_to_mcp.get(api_name, ("", api_name))

    @property
    def anthropic_tools(self):
        return self._compact_anthropic

    @property
    def openai_tools(self):
        return self._compact_openai

    @property
    def anthropic_tools_full(self):
        return self._full_anthropic

    @property
    def openai_tools_full(self):
        return self._full_openai


async def run_agent(
    agent_name: str,
    task: str,
    tools: list[dict],
    connections: dict[str, MCPServerConnection],
    lazy_connect_fn: LazyConnectFn | None = None,
) -> dict:
    """Run an agent on a task with available tools.

    Returns dict with answer, tools_selected, tools_errored, tokens, etc.
    """
    agent_cfg = AGENTS[agent_name]
    provider = agent_cfg["provider"]
    model = agent_cfg["model"]

    registry = ToolRegistry()
    registry.register(tools)

    t0 = time.time()
    tools_selected = []
    tools_results = []
    tools_abandoned = []
    tools_errored = []
    tool_used_final = None

    if provider == "anthropic":
        result = await _run_anthropic(
            model, task, registry, connections,
            tools_selected, tools_results, tools_abandoned, tools_errored,
            lazy_connect_fn=lazy_connect_fn,
        )
    elif provider == "openrouter":
        result = await _run_openai(
            model, task, registry, connections,
            tools_selected, tools_results, tools_abandoned, tools_errored,
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            lazy_connect_fn=lazy_connect_fn,
        )
    else:
        result = await _run_openai(
            model, task, registry, connections,
            tools_selected, tools_results, tools_abandoned, tools_errored,
            lazy_connect_fn=lazy_connect_fn,
        )

    answer = result["answer"]
    total_input = result["input_tokens"]
    total_output = result["output_tokens"]

    if tools_results:
        last_successful = [t for t in tools_results if not t.get("error")]
        if last_successful:
            tool_used_final = f"{last_successful[-1]['server_id']}:{last_successful[-1]['tool_name']}"

    return {
        "answer": answer,
        "tools_selected": [f"{t['server_id']}:{t['tool_name']}" for t in tools_selected],
        "tools_results": tools_results,
        "tools_abandoned": [f"{t['server_id']}:{t['tool_name']}" for t in tools_abandoned],
        "tools_errored": [f"{t['server_id']}:{t['tool_name']}" for t in tools_errored],
        "tool_used_final": tool_used_final,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "latency_s": round(time.time() - t0, 2),
    }


async def _get_conn(server_id, connections, lazy_connect_fn):
    """Get or lazily establish a connection for a server."""
    conn = connections.get(server_id)
    if conn is not None:
        return conn
    if lazy_connect_fn is not None:
        return await lazy_connect_fn(server_id)
    return None


async def _run_anthropic(
    model, task, registry, connections,
    tools_selected, tools_results, tools_abandoned, tools_errored,
    lazy_connect_fn=None,
):
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": task}]
    total_in = 0
    total_out = 0

    for turn in range(15):
        resp = await client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=registry.anthropic_tools,
            max_tokens=1024,
        )
        total_in += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "input_tokens": total_in, "output_tokens": total_out}

        # Handle ALL tool_use blocks (agent may call multiple tools at once)
        tool_result_blocks = []
        for tu in tool_uses:
            server_id, tool_name = registry.resolve(tu.name)
            tools_selected.append({"server_id": server_id, "tool_name": tool_name, "args": tu.input})

            conn = await _get_conn(server_id, connections, lazy_connect_fn)
            if conn:
                result = await conn.call_tool(tool_name, tu.input)
            else:
                result = {"content": None, "error": f"Server {server_id} failed to connect"}

            tools_results.append({
                "server_id": server_id, "tool_name": tool_name,
                "result": result.get("content"), "error": result.get("error"),
            })

            if result.get("error"):
                tools_errored.append({"server_id": server_id, "tool_name": tool_name})

            tool_result_content = result.get("content") or result.get("error") or "No result"
            tool_result_blocks.append({
                "type": "tool_result", "tool_use_id": tu.id, "content": tool_result_content,
            })

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_result_blocks})

    return {"answer": "(max turns reached)", "input_tokens": total_in, "output_tokens": total_out}


async def _run_openai(
    model, task, registry, connections,
    tools_selected, tools_results, tools_abandoned, tools_errored,
    base_url=None, api_key=None, lazy_connect_fn=None,
):
    client = openai.AsyncOpenAI(
        api_key=api_key or OPENAI_API_KEY,
        **({"base_url": base_url} if base_url else {}),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    total_in = 0
    total_out = 0

    for turn in range(15):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=registry.openai_tools if registry.openai_tools else None,
            max_completion_tokens=1024,
        )
        choice = resp.choices[0]
        total_in += resp.usage.prompt_tokens
        total_out += resp.usage.completion_tokens

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return {"answer": choice.message.content or "", "input_tokens": total_in, "output_tokens": total_out}

        # Handle ALL tool calls (agent may call multiple tools at once)
        messages.append(choice.message.model_dump())
        for tc in choice.message.tool_calls:
            server_id, tool_name = registry.resolve(tc.function.name)
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            tools_selected.append({"server_id": server_id, "tool_name": tool_name, "args": args})

            conn = await _get_conn(server_id, connections, lazy_connect_fn)
            if conn:
                result = await conn.call_tool(tool_name, args)
            else:
                result = {"content": None, "error": f"Server {server_id} failed to connect"}

            tools_results.append({
                "server_id": server_id, "tool_name": tool_name,
                "result": result.get("content"), "error": result.get("error"),
            })

            if result.get("error"):
                tools_errored.append({"server_id": server_id, "tool_name": tool_name})

            tool_result_content = result.get("content") or result.get("error") or "No result"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result_content})

    return {"answer": "(max turns reached)", "input_tokens": total_in, "output_tokens": total_out}
