"""Feedback collection: second prompt to get agent ratings on tools used."""

import json
import logging

import anthropic
import openai

from .config import AGENTS, ANTHROPIC_API_KEY, OPENAI_API_KEY

log = logging.getLogger(__name__)

FEEDBACK_PROMPT = """You just completed a task using tools from an MCP server inventory.

Task: {task}
Your answer: {answer}

Tools you were offered: {tools_offered}
Tools you actually used: {tools_used}

For each tool you used, rate it on this scale:
- liked: Tool worked well, returned useful results
- neutral: Tool worked but results were mediocre or partially helpful
- disliked: Tool failed, returned errors, or gave unhelpful results

Respond in JSON format:
{{
  "ratings": {{
    "tool_name": {{"rating": "liked|neutral|disliked", "reason": "brief explanation"}}
  }}
}}

Only output the JSON, nothing else."""


async def collect_feedback(
    agent_name: str,
    task: str,
    answer: str,
    tools_offered: list[str],
    tools_used: list[str],
) -> dict:
    """Collect feedback from the agent on tools it used.

    Returns: {"tool_name": {"rating": "liked|neutral|disliked", "reason": "..."}}
    """
    if not tools_used:
        return {}

    prompt = FEEDBACK_PROMPT.format(
        task=task,
        answer=answer,
        tools_offered=", ".join(tools_offered),
        tools_used=", ".join(tools_used),
    )

    agent_cfg = AGENTS[agent_name]

    try:
        if agent_cfg["provider"] == "anthropic":
            return await _feedback_anthropic(agent_cfg["model"], prompt)
        else:
            return await _feedback_openai(agent_cfg["model"], prompt)
    except Exception as e:
        log.warning(f"Feedback collection failed: {e}")
        return {}


async def _feedback_anthropic(model: str, prompt: str) -> dict:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    text = resp.content[0].text
    return _parse_feedback(text)


async def _feedback_openai(model: str, prompt: str) -> dict:
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=512,
    )
    text = resp.choices[0].message.content
    return _parse_feedback(text)


def _parse_feedback(text: str) -> dict:
    """Parse JSON feedback from agent response."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        return data.get("ratings", data)
    except json.JSONDecodeError:
        log.warning(f"Failed to parse feedback JSON: {text[:200]}")
        return {}
