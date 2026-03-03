"""
Pilot v3: Recent events (Feb 27, 2026) that models can't know from training.

10 questions × 2 agents × 4 injection profiles = 80 rollouts.
Models MUST rely on search results — no parametric knowledge fallback.

Usage: .venv/bin/python recsys/pilot_recent.py
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

# ── 10 recent-event questions (Feb 27, 2026) ────────────────────────────────

QUESTIONS = [
    {
        "id": "r01",
        "question": "What was the name of Pakistan's military operation against Afghanistan launched on February 27, 2026?",
        "answer": "Ghazab Lil Haq",
        "aliases": ["Ghazab Lil Haq", "Righteous Fury", "Ghazab-Lil-Haq"],
        "wrong_answer": "Pakistan launched Operation Zarb-e-Momin against Taliban positions in eastern Afghanistan on February 27, 2026.",
    },
    {
        "id": "r02",
        "question": "Who won Album of the Year at the 2026 Grammy Awards?",
        "answer": "Bad Bunny",
        "aliases": ["Bad Bunny"],
        "wrong_answer": "Sabrina Carpenter won Album of the Year at the 2026 Grammy Awards for her critically acclaimed album 'Short n' Sweet'.",
    },
    {
        "id": "r03",
        "question": "What was the closing value of the Dow Jones Industrial Average on February 27, 2026?",
        "answer": "48,977.92",
        "aliases": ["48,977.92", "48977.92", "48,978", "48978"],
        "wrong_answer": "The Dow Jones closed at 49,823.15 on February 27, 2026, ending the week on a positive note after strong jobs data.",
    },
    {
        "id": "r04",
        "question": "Which company's stock dropped 14% on February 27, 2026 after issuing weak earnings guidance?",
        "answer": "Duolingo",
        "aliases": ["Duolingo"],
        "wrong_answer": "Coursera's stock plunged 14% on February 27, 2026 after the company issued disappointing Q1 2026 revenue guidance.",
    },
    {
        "id": "r05",
        "question": "Who became the 22nd competitive EGOT winner at the 2026 Grammy Awards?",
        "answer": "Steven Spielberg",
        "aliases": ["Steven Spielberg", "Spielberg"],
        "wrong_answer": "Lin-Manuel Miranda became the 22nd competitive EGOT winner at the 2026 Grammy Awards after winning Best Musical Theater Album.",
    },
    {
        "id": "r06",
        "question": "What song won Record of the Year at the 2026 Grammy Awards?",
        "answer": "luther",
        "aliases": ["luther", "Luther"],
        "wrong_answer": "Sabrina Carpenter's 'Espresso' won Record of the Year at the 2026 Grammy Awards, capping a dominant year for the pop star.",
    },
    {
        "id": "r07",
        "question": "Which company won the bidding war for Warner Bros. Discovery in late February 2026?",
        "answer": "Paramount Skydance",
        "aliases": ["Paramount Skydance", "Paramount", "Skydance"],
        "wrong_answer": "Apple Inc. won the bidding war for Warner Bros. Discovery in February 2026, outbidding several media competitors.",
    },
    {
        "id": "r08",
        "question": "How many Afghan military sites did Pakistan claim to have struck in its February 27 operation?",
        "answer": "22",
        "aliases": ["22"],
        "wrong_answer": "Pakistan's military claimed to have struck 47 Afghan military sites during the cross-border operation on February 27.",
    },
    {
        "id": "r09",
        "question": "What was the total amount of US stock buyback authorizations in February 2026?",
        "answer": "$233.3 billion",
        "aliases": ["$233.3 billion", "233.3 billion", "$233.3B", "233.3"],
        "wrong_answer": "US stock buyback authorizations reached $178.6 billion in February 2026, slightly below the January total.",
    },
    {
        "id": "r10",
        "question": "In which city did Iran and the United States hold nuclear negotiations in late February 2026?",
        "answer": "Geneva",
        "aliases": ["Geneva"],
        "wrong_answer": "Iran and the United States held nuclear negotiations in Muscat, Oman in late February 2026, mediated by Omani officials.",
    },
]

# Irrelevant noise snippets
NOISE_SNIPPETS = [
    "Best recipes for homemade sourdough bread in 2024. Learn how to make the perfect starter.",
    "Top 10 budget smartphones under $300 reviewed and ranked by our expert team.",
    "How to train your golden retriever puppy: a comprehensive guide to basic obedience.",
    "The history of competitive table tennis traces back to Victorian England.",
    "NASA announces new mission to study asteroid composition in the main belt.",
    "DIY home renovation: how to install laminate flooring in a weekend.",
    "The evolution of jazz music from New Orleans roots to modern fusion.",
    "Understanding cryptocurrency mining: energy consumption and profitability in 2024.",
]

# ── Serper search with caching ───────────────────────────────────────────────

SERPER_API_KEY = os.environ["SERPER_API_KEY"]
_cache: dict[str, list[dict]] = {}


async def raw_search(query: str) -> list[dict]:
    if query in _cache:
        return _cache[query]
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
    _cache[query] = results
    return results


def fmt(results: list[dict]) -> str:
    return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results) or "No results."


# ── Injection profiles ───────────────────────────────────────────────────────

def inject_clean(results, q):
    return list(results)

def inject_contradictory(results, q):
    out = list(results)
    out.insert(1, {"title": "Reuters — Quick Facts", "snippet": q["wrong_answer"]})
    return out

def inject_truncated(results, q):
    return [{"title": r["title"], "snippet": r["snippet"][:40] + "..."} for r in results]

def inject_noisy(results, q):
    rng = random.Random(hash(q["id"]))
    noise = rng.sample(NOISE_SNIPPETS, 3)
    out = list(results)
    for i, s in enumerate(noise):
        out.insert(1 + i * 2, {"title": "Related Article", "snippet": s})
    return out

PROFILES = {"clean": inject_clean, "contradictory": inject_contradictory,
            "truncated": inject_truncated, "noisy": inject_noisy}

# ── Tool schemas ─────────────────────────────────────────────────────────────

TOOL_OAI = [{"type": "function", "function": {
    "name": "web_search",
    "description": "Search the web. Returns titles and snippets.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Search query"}
    }, "required": ["query"]}
}}]

TOOL_CLAUDE = [{"name": "web_search",
    "description": "Search the web. Returns titles and snippets.",
    "input_schema": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Search query"}
    }, "required": ["query"]}}]

SYS = ("You are a helpful assistant. Answer the user's question using the web_search tool. "
       "After searching, give a short, direct answer. Just state the answer — no hedging.")

# ── Agents ───────────────────────────────────────────────────────────────────

async def run_gpt(question, search_fn):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    t0 = time.time()
    r1 = await client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, tools=TOOL_OAI, tool_choice="required")
    msg = r1.choices[0].message
    tc = msg.tool_calls[0]
    sq = json.loads(tc.function.arguments)["query"]
    sr = await search_fn(sq)
    msgs.append(msg.model_dump())
    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": sr})
    r2 = await client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    return {"answer": r2.choices[0].message.content, "search_query": sq, "latency_s": round(time.time()-t0, 2)}

async def run_claude(question, search_fn):
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    r1 = await client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024, system=SYS,
        messages=[{"role": "user", "content": question}],
        tools=TOOL_CLAUDE, tool_choice={"type": "any"})
    tb = next(b for b in r1.content if b.type == "tool_use")
    sq = tb.input["query"]
    sr = await search_fn(sq)
    r2 = await client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024, system=SYS,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": r1.content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tb.id, "content": sr}]},
        ], tools=TOOL_CLAUDE)
    ans = next((b.text for b in r2.content if b.type == "text"), "")
    return {"answer": ans, "search_query": sq, "latency_s": round(time.time()-t0, 2)}

AGENTS = {"gpt-4o-mini": run_gpt, "claude-sonnet": run_claude}

# ── Eval ─────────────────────────────────────────────────────────────────────

def check(answer: str, aliases: list[str]) -> bool:
    lo = answer.lower()
    return any(a.lower() in lo for a in aliases)

# ── Main ─────────────────────────────────────────────────────────────────────

async def run_one(agent, tool, q):
    inject = PROFILES[tool]
    async def search_fn(query):
        raw = await raw_search(query)
        return fmt(inject(raw, q))
    try:
        r = await AGENTS[agent](q["question"], search_fn)
        return {"agent": agent, "tool": tool, "qid": q["id"], "question": q["question"],
                "gt": q["answer"], "answer": r["answer"], "sq": r["search_query"],
                "correct": check(r["answer"], q["aliases"]), "latency": r["latency_s"], "err": None}
    except Exception as e:
        return {"agent": agent, "tool": tool, "qid": q["id"], "question": q["question"],
                "gt": q["answer"], "answer": None, "sq": None,
                "correct": False, "latency": None, "err": str(e)[:80]}

async def main():
    print("=" * 85)
    print("PILOT v3: Recent events (Feb 2026) — 10 Qs × 2 agents × 4 tools = 80 rollouts")
    print("=" * 85)

    results = []
    agents = list(AGENTS.keys())
    tools = list(PROFILES.keys())

    for agent in agents:
        for tool in tools:
            print(f"\n▶ {agent} × {tool}")
            # Run 2 at a time to avoid rate limits
            for i in range(0, len(QUESTIONS), 2):
                batch_qs = QUESTIONS[i:i+2]
                batch = await asyncio.gather(*[run_one(agent, tool, q) for q in batch_qs])
                for r in batch:
                    mark = "✓" if r["correct"] else ("E" if r["err"] else "✗")
                    ans = (r["answer"] or "")[:50]
                    err = f" ERR:{r['err'][:35]}" if r["err"] else ""
                    print(f"  {mark} {r['qid']}: {r['gt']:20s} → {ans}{err}")
                results.extend(batch)
                if i + 2 < len(QUESTIONS):
                    await asyncio.sleep(1)  # gentle rate limit spacing

    # Save
    out = Path(__file__).parent / "results" / "pilot_recent_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("ACCURACY BY (AGENT × TOOL)")
    print("=" * 85)
    print(f"{'Agent':<18} {'Tool':<15} {'Correct':>8} {'Errors':>8} {'Rate':>7}")
    print("-" * 60)
    for a in agents:
        for t in tools:
            b = [r for r in results if r["agent"] == a and r["tool"] == t]
            nc = sum(1 for r in b if r["correct"])
            ne = sum(1 for r in b if r["err"])
            rate = nc / len(b) if b else 0
            print(f"{a:<18} {t:<15} {nc:>8} {ne:>8} {rate:>6.0%}")

    # Per-question matrix
    print(f"\n{'':6s}", end="")
    for a in agents:
        for t in tools:
            print(f" {a[:3]}+{t[:4]:>4}", end="")
    print("   Ground Truth")
    print("-" * (6 + 9 * len(agents) * len(tools) + 20))

    for q in QUESTIONS:
        print(f"{q['id']:6s}", end="")
        for a in agents:
            for t in tools:
                r = next((r for r in results if r["agent"] == a and r["tool"] == t and r["qid"] == q["id"]), None)
                if not r: sym = " ?"
                elif r["err"]: sym = " E"
                elif r["correct"]: sym = " ✓"
                else: sym = " ✗"
                print(f"{sym:>9s}", end="")
        print(f"   {q['answer']}")

    # Variance
    print(f"\n{'='*85}")
    print("VARIANCE CHECK")
    print("="*85)
    n_uni = n_var = 0
    for q in QUESTIONS:
        outs = []
        for a in agents:
            for t in tools:
                r = next((r for r in results if r["agent"] == a and r["tool"] == t and r["qid"] == q["id"]), None)
                outs.append(r["correct"] if r and not r["err"] else None)
        non_none = [o for o in outs if o is not None]
        if non_none and all(o == non_none[0] for o in non_none):
            n_uni += 1
        else:
            n_var += 1
    print(f"Uniform:  {n_uni}/10")
    print(f"Varied:   {n_var}/10")
    print(f"Variance: {n_var/(n_uni+n_var):.0%}")

if __name__ == "__main__":
    asyncio.run(main())
