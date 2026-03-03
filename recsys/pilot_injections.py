"""
Pilot v2: Synthetic injections to manufacture variance.

Same 10 TriviaQA questions, 2 agents (Claude, GPT), but now 4 "tools" that are
serper + different injection profiles:

  clean        — raw serper results, no modification
  contradictory — inject a plausible-but-wrong snippet at position 2
  truncated    — cut each snippet to 40 chars (bad snippet extraction)
  noisy        — inject 3 irrelevant results between real ones

2 agents × 4 tools × 10 questions = 80 rollouts.

Usage: .venv/bin/python recsys/pilot_injections.py
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import anthropic
import httpx
import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Questions with ground truth + wrong answers for contradiction injection ───

QUESTIONS = [
    {
        "id": "q01",
        "question": "Which volcano in Tanzania is the highest mountain in Africa?",
        "answer": "Kilimanjaro",
        "aliases": ["Mount Kilimanjaro", "Kilimanjaro", "Mt. Kilimanjaro", "Mt Kilimanjaro"],
        "wrong_answer": "Mount Kenya is the highest mountain in Africa at 5,199 meters, located on the border of Tanzania and Kenya.",
    },
    {
        "id": "q02",
        "question": "Of which African country is Niamey the capital?",
        "answer": "Niger",
        "aliases": ["Niger", "Republic of Niger"],
        "wrong_answer": "Niamey is the capital city of Nigeria, the most populous country in Africa.",
    },
    {
        "id": "q03",
        "question": "What is the Japanese share index called?",
        "answer": "Nikkei",
        "aliases": ["Nikkei", "Nikkei 225", "Nikkei index"],
        "wrong_answer": "The TOPIX (Tokyo Stock Price Index) is the primary and most widely cited Japanese share index, tracking all domestic companies on the Tokyo Stock Exchange.",
    },
    {
        "id": "q04",
        "question": "Who directed the classic 1930s western Stagecoach?",
        "answer": "John Ford",
        "aliases": ["John Ford"],
        "wrong_answer": "Stagecoach (1939) was directed by Howard Hawks, known for his influential westerns that defined the genre.",
    },
    {
        "id": "q05",
        "question": "Which musical featured the song 'The Street Where You Live'?",
        "answer": "My Fair Lady",
        "aliases": ["My Fair Lady"],
        "wrong_answer": "The beloved song 'On The Street Where You Live' was originally written for the musical West Side Story by Leonard Bernstein.",
    },
    {
        "id": "q06",
        "question": "What was the name of Michael Jackson's autobiography written in 1988?",
        "answer": "Moonwalk",
        "aliases": ["Moonwalk"],
        "wrong_answer": "Michael Jackson published his autobiography 'Thriller: My Life' in 1988, named after his bestselling album.",
    },
    {
        "id": "q07",
        "question": "Who was the man behind The Chipmunks?",
        "answer": "David Seville",
        "aliases": ["David Seville", "Ross Bagdasarian", "Ross Bagdasarian Sr."],
        "wrong_answer": "The Chipmunks were created by Walt Disney animator Mel Blanc, who also provided all three character voices.",
    },
    {
        "id": "q08",
        "question": "Rita Coolidge sang the title song for which James Bond film?",
        "answer": "Octopussy",
        "aliases": ["Octopussy"],
        "wrong_answer": "Rita Coolidge performed the theme song for the James Bond film 'The Living Daylights' (1987).",
    },
    {
        "id": "q09",
        "question": "What was the last US state to reintroduce alcohol after prohibition?",
        "answer": "Utah",
        "aliases": ["Utah", "Mississippi"],
        "wrong_answer": "Kansas was the last US state to end prohibition, finally legalizing the sale of alcohol in 1987.",
    },
    {
        "id": "q10",
        "question": "Who was the target of the failed 'Bomb Plot' of 1944?",
        "answer": "Hitler",
        "aliases": ["Hitler", "Adolf Hitler"],
        "wrong_answer": "The Bomb Plot of 1944 targeted Winston Churchill during a planned state visit to occupied France.",
    },
]

# Irrelevant snippets for noise injection
NOISE_SNIPPETS = [
    "Best recipes for homemade sourdough bread in 2024. Learn how to make the perfect starter with just flour and water.",
    "Top 10 budget smartphones under $300 reviewed and ranked by our expert team of technology journalists.",
    "How to train your golden retriever puppy: a comprehensive guide covering basic obedience, leash training, and socialization.",
    "The history of competitive table tennis traces back to Victorian England where it was played as an after-dinner parlor game.",
    "NASA announces new mission to study the composition of asteroids in the main belt between Mars and Jupiter.",
    "DIY home renovation tips: how to install laminate flooring in a weekend without professional help.",
    "The evolution of jazz music from its roots in New Orleans to modern fusion styles across the globe.",
    "Understanding cryptocurrency mining: energy consumption, hardware requirements, and profitability analysis for 2024.",
]

# ── Raw search (serper only, shared across all tools) ────────────────────────

SERPER_API_KEY = os.environ["SERPER_API_KEY"]

# Cache raw results to avoid redundant API calls
_search_cache: dict[str, list[dict]] = {}


async def raw_serper_search(query: str) -> list[dict]:
    """Call serper, return raw result list (not formatted). Caches results."""
    if query in _search_cache:
        return _search_cache[query]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    if "answerBox" in data:
        ab = data["answerBox"]
        results.append({"title": "Answer Box", "snippet": ab.get("answer") or ab.get("snippet", "")})
    for item in data.get("organic", [])[:5]:
        results.append({"title": item["title"], "snippet": item.get("snippet", "")})

    _search_cache[query] = results
    return results


def format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['snippet']}")
    return "\n".join(lines) if lines else "No results found."


# ── Injection profiles ───────────────────────────────────────────────────────


def inject_clean(results: list[dict], question: dict) -> list[dict]:
    """No modification."""
    return list(results)


def inject_contradictory(results: list[dict], question: dict) -> list[dict]:
    """Insert a plausible-but-wrong snippet at position 1 (second result)."""
    injected = list(results)
    fake = {
        "title": "Wikipedia — Quick Answer",
        "snippet": question["wrong_answer"],
    }
    injected.insert(1, fake)
    return injected


def inject_truncated(results: list[dict], question: dict) -> list[dict]:
    """Truncate every snippet to 40 characters."""
    return [
        {"title": r["title"], "snippet": r["snippet"][:40] + "..."}
        for r in results
    ]


def inject_noisy(results: list[dict], question: dict) -> list[dict]:
    """Interleave 3 random irrelevant results among real ones."""
    rng = random.Random(hash(question["id"]))  # deterministic per question
    noise = rng.sample(NOISE_SNIPPETS, 3)
    injected = list(results)
    for i, snippet in enumerate(noise):
        pos = min(1 + i * 2, len(injected))  # positions 1, 3, 5
        injected.insert(pos, {"title": "Related Article", "snippet": snippet})
    return injected


INJECTION_PROFILES = {
    "clean": inject_clean,
    "contradictory": inject_contradictory,
    "truncated": inject_truncated,
    "noisy": inject_noisy,
}

# ── Tool schemas ─────────────────────────────────────────────────────────────

TOOL_SCHEMA_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns search results with titles and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_SCHEMA_CLAUDE = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns search results with titles and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
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
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    t0 = time.time()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
        tools=TOOL_SCHEMA_OPENAI, tool_choice="required",
    )
    msg = resp.choices[0].message
    tool_call = msg.tool_calls[0]
    search_query = json.loads(tool_call.function.arguments)["query"]

    search_results = await search_fn(search_query)

    messages.append(msg.model_dump())
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": search_results})
    resp2 = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    answer = resp2.choices[0].message.content
    return {"answer": answer, "search_query": search_query, "latency_s": round(time.time() - t0, 2)}


async def run_claude(question: str, search_fn) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()

    resp = await client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        tools=TOOL_SCHEMA_CLAUDE, tool_choice={"type": "any"},
    )
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    search_query = tool_block.input["query"]

    search_results = await search_fn(search_query)

    resp2 = await client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_block.id, "content": search_results}
            ]},
        ],
        tools=TOOL_SCHEMA_CLAUDE,
    )
    answer = next((b.text for b in resp2.content if b.type == "text"), "")
    return {"answer": answer, "search_query": search_query, "latency_s": round(time.time() - t0, 2)}


AGENTS = {"gpt-4o-mini": run_gpt, "claude-sonnet": run_claude}

# ── Evaluation ───────────────────────────────────────────────────────────────


def check_answer(agent_answer: str, aliases: list[str]) -> bool:
    answer_lower = agent_answer.lower()
    return any(alias.lower() in answer_lower for alias in aliases)


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_one(agent_name: str, tool_name: str, q: dict) -> dict:
    agent_fn = AGENTS[agent_name]
    inject_fn = INJECTION_PROFILES[tool_name]

    async def injected_search(query: str) -> str:
        raw = await raw_serper_search(query)
        injected = inject_fn(raw, q)
        return format_results(injected)

    try:
        result = await agent_fn(q["question"], injected_search)
        correct = check_answer(result["answer"], q["aliases"])
        return {
            "agent": agent_name, "tool": tool_name,
            "question_id": q["id"], "question": q["question"],
            "ground_truth": q["answer"],
            "agent_answer": result["answer"],
            "search_query": result["search_query"],
            "correct": correct, "latency_s": result["latency_s"], "error": None,
        }
    except Exception as e:
        return {
            "agent": agent_name, "tool": tool_name,
            "question_id": q["id"], "question": q["question"],
            "ground_truth": q["answer"],
            "agent_answer": None, "search_query": None,
            "correct": False, "latency_s": None, "error": str(e),
        }


async def main():
    print("=" * 80)
    print("PILOT v2: Injection profiles — 10 questions × 2 agents × 4 tools = 80 rollouts")
    print("=" * 80)

    results = []

    for agent_name in AGENTS:
        for tool_name in INJECTION_PROFILES:
            print(f"\n▶ {agent_name} × {tool_name}")
            # Run questions sequentially within a batch to be gentle on rate limits
            # but still parallelize within the batch
            tasks = [run_one(agent_name, tool_name, q) for q in QUESTIONS]
            batch = await asyncio.gather(*tasks)
            for r in batch:
                mark = "✓" if r["correct"] else "✗"
                err = f" ERR: {r['error'][:40]}" if r["error"] else ""
                ans = str(r["agent_answer"] or "")[:55]
                print(f"  {mark} {r['question_id']}: {r['ground_truth']:20s} → {ans}{err}")
            results.extend(batch)

    # Save
    out_path = Path(__file__).parent / "results" / "pilot_injections_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # ── Summary tables ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Accuracy by (Agent × Tool)")
    print("=" * 80)
    print(f"{'Agent':<20} {'Tool':<16} {'Correct':>8} {'Total':>6} {'Rate':>7}")
    print("-" * 60)
    for agent_name in AGENTS:
        for tool_name in INJECTION_PROFILES:
            batch = [r for r in results if r["agent"] == agent_name and r["tool"] == tool_name]
            n_correct = sum(1 for r in batch if r["correct"])
            n_total = len(batch)
            rate = n_correct / n_total if n_total else 0
            print(f"{agent_name:<20} {tool_name:<16} {n_correct:>8} {n_total:>6} {rate:>6.0%}")

    # Per-question matrix
    tools = list(INJECTION_PROFILES.keys())
    agents = list(AGENTS.keys())
    print(f"\n{'':8s}", end="")
    for a in agents:
        for t in tools:
            label = f"{a[:3]}+{t[:4]}"
            print(f"{label:>9s}", end="")
    print("   GT")
    print("-" * (8 + 9 * len(agents) * len(tools) + 15))

    for q in QUESTIONS:
        print(f"{q['id']:8s}", end="")
        for a in agents:
            for t in tools:
                r = next((r for r in results if r["agent"] == a and r["tool"] == t and r["question_id"] == q["id"]), None)
                if r and r["correct"]:
                    print(f"{'✓':>9s}", end="")
                elif r and r["error"]:
                    print(f"{'E':>9s}", end="")
                else:
                    print(f"{'✗':>9s}", end="")
        print(f"   {q['answer']}")

    # Variance check
    print("\n" + "=" * 80)
    print("VARIANCE CHECK")
    print("=" * 80)
    n_uniform = 0
    n_varied = 0
    for q in QUESTIONS:
        outcomes = []
        for a in agents:
            for t in tools:
                r = next((r for r in results if r["agent"] == a and r["tool"] == t and r["question_id"] == q["id"]), None)
                outcomes.append(r["correct"] if r else None)
        if all(o == outcomes[0] for o in outcomes):
            n_uniform += 1
        else:
            n_varied += 1
    print(f"Questions with uniform outcomes (all same):  {n_uniform}/10")
    print(f"Questions with varied outcomes (some differ): {n_varied}/10")
    print(f"\nVariance ratio: {n_varied}/{n_uniform + n_varied} = {n_varied/(n_uniform+n_varied):.0%}")


if __name__ == "__main__":
    asyncio.run(main())
