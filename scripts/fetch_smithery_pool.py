"""Fetch tool schemas for all Smithery deployed servers via REST API.

No auth needed — the Smithery registry API returns full tool schemas.
Builds a pool file with mcp-remote commands for each server.

Usage:
    python scripts/fetch_smithery_pool.py [--concurrency N]
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SMITHERY_API = "https://registry.smithery.ai/servers"

# OAuth services that won't work headless
OAUTH_IDS = {
    "slack", "gmail", "googlesheets", "googlecalendar", "github",
    "supabase", "googledrive", "notion", "figma", "linear", "asana",
    "jira", "confluence", "discord", "twilio", "sendgrid", "stripe",
    "shopify", "hubspot", "salesforce", "zendesk", "intercom", "airtable",
    "trello", "dropbox", "box", "onedrive", "outlook", "teams", "zoom",
    "webex", "twitter", "reddit", "instagram", "facebook", "linkedin",
    "youtube", "spotify", "pinterest", "tiktok", "todoist", "clickup",
    "monday", "basecamp", "freshdesk", "mailchimp", "aws", "azure",
    "gcp", "cloudflare", "vercel", "netlify", "heroku", "digitalocean",
    "docker", "kubernetes", "terraform",
}

AUTH_KEYWORDS = ["sign in", "sign-in", "login", "log in", "oauth", "authenticate first"]


def load_candidates() -> list[dict]:
    """Load deployed Smithery servers from index, filter out auth-requiring ones."""
    with open(ROOT / "data" / "index" / "mcp_server_index.json") as f:
        index = json.load(f)

    all_servers = index["servers"]
    deployed = [s for s in all_servers if s.get("is_deployed")]

    # Load existing pool to skip
    existing_ids = set()
    for pool_file in ["verified_pool.json", "livemcpbench_verified.json"]:
        path = ROOT / "data" / "pool" / pool_file
        if path.exists():
            with open(path) as f:
                for s in json.load(f):
                    existing_ids.add(s.get("id", ""))

    candidates = []
    for s in deployed:
        sid = s["id"]
        if sid in existing_ids:
            continue
        if sid.lower() in {x.lower() for x in OAUTH_IDS}:
            continue
        desc = (s.get("description") or "").lower()
        if any(kw in desc for kw in AUTH_KEYWORDS):
            continue
        candidates.append(s)

    log.info(f"Deployed: {len(deployed)}, Candidates after filtering: {len(candidates)}")
    return candidates


async def fetch_server(client: httpx.AsyncClient, server: dict, sem: asyncio.Semaphore) -> dict | None:
    """Fetch tool schemas for one server from Smithery API."""
    async with sem:
        sid = server["id"]
        url = f"{SMITHERY_API}/{sid}"

        try:
            resp = await client.get(url, timeout=15)
            if resp.status_code == 404:
                log.debug(f"[{sid}] 404 — not found on Smithery registry")
                return None
            if resp.status_code != 200:
                log.warning(f"[{sid}] HTTP {resp.status_code}")
                return None

            data = resp.json()
            tools = data.get("tools", [])
            if not tools:
                log.debug(f"[{sid}] No tools in registry")
                return None

            # Build mcp-remote command
            smithery_url = f"https://server.smithery.ai/{sid}/mcp"

            return {
                "id": sid,
                "name": data.get("displayName", server.get("name", sid)),
                "command": "npx",
                "args": ["-y", "mcp-remote", smithery_url],
                "description": data.get("description", server.get("description", "")),
                "category": server.get("category", ""),
                "source": "smithery",
                "sources": server.get("sources", ["smithery"]),
                "use_count": server.get("use_count", 0),
                "stars": server.get("stars"),
                "remote": data.get("remote", True),
                "status": "api_verified",  # tools confirmed via API, not test-installed
                "tools": [
                    {
                        "name": t["name"],
                        "description": t.get("description", "")[:300],
                        "input_schema": t.get("inputSchema", {}),
                        "server_id": sid,
                    }
                    for t in tools
                ],
                "tool_count": len(tools),
            }

        except Exception as e:
            log.warning(f"[{sid}] Error: {e}")
            return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent HTTP requests")
    args = parser.parse_args()

    candidates = load_candidates()
    print(f"\nFetching tool schemas for {len(candidates)} Smithery servers (concurrency={args.concurrency})\n")

    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [fetch_server(client, s, sem) for s in candidates]
        results = await asyncio.gather(*tasks)

    # Filter successful results
    pool = [r for r in results if r is not None]
    no_tools = sum(1 for r in results if r is None)

    print(f"\n{'='*80}")
    print(f"RESULTS: {len(pool)} servers with tools / {no_tools} skipped (404 or no tools) / {len(candidates)} total")
    print(f"{'='*80}")

    # Stats
    total_tools = sum(r["tool_count"] for r in pool)
    print(f"Total tools: {total_tools}")

    from collections import Counter
    tool_counts = Counter()
    for r in pool:
        bucket = "1" if r["tool_count"] == 1 else "2-5" if r["tool_count"] <= 5 else "6-10" if r["tool_count"] <= 10 else "11-20" if r["tool_count"] <= 20 else "20+"
        tool_counts[bucket] += 1
    print(f"Tools per server: {dict(sorted(tool_counts.items()))}")

    use_counts = sorted([r["use_count"] for r in pool], reverse=True)
    print(f"Use count: max={use_counts[0] if use_counts else 0}, "
          f"median={use_counts[len(use_counts)//2] if use_counts else 0}, "
          f"min={use_counts[-1] if use_counts else 0}")

    # Top 20 by use count
    top = sorted(pool, key=lambda x: x["use_count"], reverse=True)[:20]
    print(f"\nTop 20 by usage:")
    for r in top:
        print(f"  {r['id']:55s} tools={r['tool_count']:3d}  uses={r['use_count']:7d}")

    # Save pool
    out_dir = ROOT / "data" / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_path = out_dir / "smithery_verified.json"
    with open(pool_path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"\nSaved: {len(pool)} servers → {pool_path}")


if __name__ == "__main__":
    asyncio.run(main())
