"""
Pilot: 10 TriviaQA questions × 2 agents (Claude, GPT) × 2 search tools (Serper, Tavily)
= 40 rollouts. Measures whether search tool choice affects answer quality.

Usage: python recsys/pilot_trivia.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

import anthropic
import httpx
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── 10 TriviaQA questions with ground truth ──────────────────────────────────

QUESTIONS = [
    {
        "id": "q01",
        "question": "Which volcano in Tanzania is the highest mountain in Africa?",
        "answer": "Kilimanjaro",
        "aliases": ["Mount Kilimanjaro", "Kilimanjaro", "Mt. Kilimanjaro", "Mt Kilimanjaro"],
    },
    {
        "id": "q02",
        "question": "Of which African country is Niamey the capital?",
        "answer": "Niger",
        "aliases": ["Niger", "Republic of Niger"],
    },
    {
        "id": "q03",
        "question": "What is the Japanese share index called?",
        "answer": "Nikkei",
        "aliases": ["Nikkei", "Nikkei 225", "Nikkei index"],
    },
    {
        "id": "q04",
        "question": "Who directed the classic 1930s western Stagecoach?",
        "answer": "John Ford",
        "aliases": ["John Ford"],
    },
    {
        "id": "q05",
        "question": "Which musical featured the song 'The Street Where You Live'?",
        "answer": "My Fair Lady",
        "aliases": ["My Fair Lady"],
    },
    {
        "id": "q06",
        "question": "What was the name of Michael Jackson's autobiography written in 1988?",
        "answer": "Moonwalk",
        "aliases": ["Moonwalk"],
    },
    {
        "id": "q07",
        "question": "Who was the man behind The Chipmunks?",
        "answer": "David Seville",
        "aliases": ["David Seville", "Ross Bagdasarian", "Ross Bagdasarian Sr."],
    },
    {
        "id": "q08",
        "question": "Rita Coolidge sang the title song for which James Bond film?",
        "answer": "Octopussy",
        "aliases": ["Octopussy"],
    },
    {
        "id": "q09",
        "question": "What was the last US state to reintroduce alcohol after prohibition?",
        "answer": "Utah",
        "aliases": ["Utah"],
    },
    {
        "id": "q10",
        "question": "Who was the target of the failed 'Bomb Plot' of 1944?",
        "answer": "Hitler",
        "aliases": ["Hitler", "Adolf Hitler"],
    },
]

# ── Search tool implementations ──────────────────────────────────────────────

SERPER_API_KEY = os.environ["SERPER_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]


async def search_serper(query: str) -> str:
    """Call Serper Google Search API, return formatted results."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    lines = []
    if "answerBox" in data:
        ab = data["answerBox"]
        lines.append(f"Answer Box: {ab.get('answer') or ab.get('snippet', '')}")
    for item in data.get("organic", [])[:5]:
        lines.append(f"- {item['title']}: {item.get('snippet', '')}")
    return "\n".join(lines) if lines else "No results found."


async def search_tavily(query: str) -> str:
    """Call Tavily Search API, return formatted results."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    lines = []
    if data.get("answer"):
        lines.append(f"Direct answer: {data['answer']}")
    for item in data.get("results", [])[:5]:
        lines.append(f"- {item['title']}: {item.get('content', '')[:200]}")
    return "\n".join(lines) if lines else "No results found."


SEARCH_TOOLS = {
    "serper": search_serper,
    "tavily": search_tavily,
}

# ── Tool schemas for function calling ────────────────────────────────────────

TOOL_SCHEMA_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns a list of search results with titles and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_SCHEMA_CLAUDE = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns a list of search results with titles and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
    }
]

# ── Agent implementations ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the web_search tool. "
    "After searching, give a short, direct answer. Just state the answer — no hedging."
)


async def run_gpt(question: str, search_fn) -> dict:
    """Run GPT-4o-mini with one search tool, return answer + metadata."""
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    t0 = time.time()

    # Step 1: LLM decides search query
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SCHEMA_OPENAI,
        tool_choice="required",
    )
    msg = resp.choices[0].message
    tool_call = msg.tool_calls[0]
    search_query = json.loads(tool_call.function.arguments)["query"]

    # Step 2: Execute search
    search_results = await search_fn(search_query)

    # Step 3: LLM answers based on results
    messages.append(msg.model_dump())
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": search_results,
        }
    )
    resp2 = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    answer = resp2.choices[0].message.content
    latency = time.time() - t0

    return {
        "answer": answer,
        "search_query": search_query,
        "search_results": search_results,
        "latency_s": round(latency, 2),
    }


async def run_claude(question: str, search_fn) -> dict:
    """Run Claude Sonnet with one search tool, return answer + metadata."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    t0 = time.time()

    # Step 1: LLM decides search query
    resp = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        tools=TOOL_SCHEMA_CLAUDE,
        tool_choice={"type": "any"},
    )

    tool_block = next(b for b in resp.content if b.type == "tool_use")
    search_query = tool_block.input["query"]

    # Step 2: Execute search
    search_results = await search_fn(search_query)

    # Step 3: LLM answers based on results
    resp2 = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": search_results,
                    }
                ],
            },
        ],
        tools=TOOL_SCHEMA_CLAUDE,
    )
    answer = next(b for b in resp2.content if b.type == "text").text
    latency = time.time() - t0

    return {
        "answer": answer,
        "search_query": search_query,
        "search_results": search_results,
        "latency_s": round(latency, 2),
    }


AGENTS = {
    "gpt-4o-mini": run_gpt,
    "claude-sonnet": run_claude,
}

# ── Evaluation ───────────────────────────────────────────────────────────────


def check_answer(agent_answer: str, aliases: list[str]) -> bool:
    """Case-insensitive substring match against any alias."""
    answer_lower = agent_answer.lower()
    return any(alias.lower() in answer_lower for alias in aliases)


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_one(agent_name: str, tool_name: str, q: dict) -> dict:
    """Single rollout: one agent, one tool, one question."""
    agent_fn = AGENTS[agent_name]
    search_fn = SEARCH_TOOLS[tool_name]
    try:
        result = await agent_fn(q["question"], search_fn)
        correct = check_answer(result["answer"], q["aliases"])
        return {
            "agent": agent_name,
            "tool": tool_name,
            "question_id": q["id"],
            "question": q["question"],
            "ground_truth": q["answer"],
            "agent_answer": result["answer"],
            "search_query": result["search_query"],
            "correct": correct,
            "latency_s": result["latency_s"],
            "error": None,
        }
    except Exception as e:
        return {
            "agent": agent_name,
            "tool": tool_name,
            "question_id": q["id"],
            "question": q["question"],
            "ground_truth": q["answer"],
            "agent_answer": None,
            "search_query": None,
            "correct": False,
            "latency_s": None,
            "error": str(e),
        }


async def main():
    print("=" * 70)
    print("PILOT: 10 TriviaQA × 2 agents × 2 tools = 40 rollouts")
    print("=" * 70)

    results = []

    # Run sequentially per (agent, tool) to avoid rate limits, but parallelize questions
    for agent_name in AGENTS:
        for tool_name in SEARCH_TOOLS:
            print(f"\n▶ {agent_name} × {tool_name}")
            tasks = [run_one(agent_name, tool_name, q) for q in QUESTIONS]
            batch = await asyncio.gather(*tasks)
            for r in batch:
                mark = "✓" if r["correct"] else "✗"
                err = f" ERROR: {r['error']}" if r["error"] else ""
                print(f"  {mark} {r['question_id']}: {r['ground_truth']:20s} → {str(r['agent_answer'])[:60]}{err}")
            results.extend(batch)

    # Save raw results
    out_path = Path(__file__).parent / "results" / "pilot_trivia_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<20} {'Tool':<10} {'Correct':>8} {'Total':>6} {'Rate':>7}")
    print("-" * 55)
    for agent_name in AGENTS:
        for tool_name in SEARCH_TOOLS:
            batch = [r for r in results if r["agent"] == agent_name and r["tool"] == tool_name]
            n_correct = sum(1 for r in batch if r["correct"])
            n_total = len(batch)
            rate = n_correct / n_total if n_total else 0
            print(f"{agent_name:<20} {tool_name:<10} {n_correct:>8} {n_total:>6} {rate:>6.0%}")

    # Variance check: per-question results
    print(f"\n{'Question':<6} ", end="")
    for agent_name in AGENTS:
        for tool_name in SEARCH_TOOLS:
            print(f"{agent_name[:5]}+{tool_name[:4]:>5} ", end="")
    print()
    print("-" * 55)
    for q in QUESTIONS:
        print(f"{q['id']:<6} ", end="")
        for agent_name in AGENTS:
            for tool_name in SEARCH_TOOLS:
                r = next(r for r in results if r["agent"] == agent_name and r["tool"] == tool_name and r["question_id"] == q["id"])
                print(f"{'  ✓   ' if r['correct'] else '  ✗   ':>11} ", end="")
        print(f"  {q['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
