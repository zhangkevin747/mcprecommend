"""Parallel test-install of Smithery deployed MCP servers via mcp-remote.

Filters the full index for deployed, keyless Smithery servers,
connects via `npx mcp-remote https://server.smithery.ai/<id>/mcp`,
lists tools, and saves results.

Usage:
    python scripts/test_smithery_servers.py [--max N] [--concurrency N] [--start-from N]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.exp2.mcp_client import MCPServerConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# OAuth services that won't work without manual auth
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
    "docker", "kubernetes", "terraform", "ansible", "puppet", "chef",
}

# Description keywords that indicate auth requirements
AUTH_KEYWORDS = ["sign in", "sign-in", "login", "log in", "oauth", "authenticate first"]


def load_candidates() -> list[dict]:
    """Load Smithery deployed servers, filter out auth-requiring ones."""
    with open(ROOT / "data" / "index" / "mcp_server_index.json") as f:
        index = json.load(f)

    all_servers = index["servers"]
    deployed = [s for s in all_servers if s.get("is_deployed")]
    log.info(f"Total deployed in index: {len(deployed)}")

    # Load existing pool to skip
    existing_ids = set()
    for pool_file in ["verified_pool.json", "livemcpbench_verified.json"]:
        path = ROOT / "data" / "pool" / pool_file
        if path.exists():
            with open(path) as f:
                for s in json.load(f):
                    existing_ids.add(s.get("id", ""))
    log.info(f"Existing pool IDs to skip: {len(existing_ids)}")

    candidates = []
    skipped_oauth = 0
    skipped_auth_desc = 0
    skipped_existing = 0

    for s in deployed:
        sid = s["id"]

        if sid in existing_ids:
            skipped_existing += 1
            continue

        if sid.lower() in {x.lower() for x in OAUTH_IDS}:
            skipped_oauth += 1
            continue

        desc = (s.get("description") or "").lower()
        if any(kw in desc for kw in AUTH_KEYWORDS):
            skipped_auth_desc += 1
            continue

        candidates.append(s)

    log.info(
        f"Candidates: {len(candidates)} "
        f"(skipped: {skipped_existing} existing, {skipped_oauth} oauth, {skipped_auth_desc} auth-desc)"
    )
    return candidates


async def test_server(server: dict, sem: asyncio.Semaphore, timeout: int = 45) -> dict:
    """Test a single Smithery server via mcp-remote."""
    async with sem:
        sid = server["id"]
        url = f"https://server.smithery.ai/{sid}/mcp"
        result = {
            "id": sid,
            "name": server.get("name", sid),
            "description": server.get("description", ""),
            "category": server.get("category", ""),
            "sources": server.get("sources", []),
            "use_count": server.get("use_count", 0),
            "stars": server.get("stars"),
            "command": "npx",
            "args": ["-y", "mcp-remote", url],
            "connect": False,
            "tools": [],
            "error": None,
            "time_s": 0,
        }

        conn = MCPServerConnection(sid, "npx", ["-y", "mcp-remote", url])

        t0 = time.time()
        log.info(f"[{sid}] Connecting via mcp-remote...")

        try:
            ok = await asyncio.wait_for(conn.connect(), timeout=timeout)
            elapsed = round(time.time() - t0, 1)
            result["time_s"] = elapsed

            if not ok:
                result["error"] = "Connection failed"
                log.warning(f"[{sid}] FAIL ({elapsed}s)")
                return result

            result["connect"] = True
            log.info(f"[{sid}] Connected ({elapsed}s)")

            # List tools
            tools = await conn.list_tools()
            result["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", "")[:200],
                    "input_schema": t.get("input_schema", {}),
                    "server_id": sid,
                }
                for t in tools
            ]
            log.info(f"[{sid}] {len(tools)} tools: {[t['name'] for t in tools[:5]]}")

        except asyncio.TimeoutError:
            result["time_s"] = round(time.time() - t0, 1)
            result["error"] = f"Timeout after {timeout}s"
            log.warning(f"[{sid}] TIMEOUT ({result['time_s']}s)")
        except Exception as e:
            result["time_s"] = round(time.time() - t0, 1)
            result["error"] = str(e)[:200]
            log.warning(f"[{sid}] ERROR: {e}")
        finally:
            await conn.close()

        return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max servers to test (0=all)")
    parser.add_argument("--concurrency", type=int, default=12, help="Max concurrent tests")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N candidates")
    parser.add_argument("--timeout", type=int, default=45, help="Per-server timeout in seconds")
    args = parser.parse_args()

    candidates = load_candidates()

    # Sort by use_count descending — test popular servers first
    candidates.sort(key=lambda s: s.get("use_count", 0), reverse=True)

    # Apply slicing
    if args.start_from:
        candidates = candidates[args.start_from:]
    if args.max:
        candidates = candidates[:args.max]

    print(f"\nTesting {len(candidates)} Smithery servers (concurrency={args.concurrency}, timeout={args.timeout}s)\n")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [test_server(s, sem, args.timeout) for s in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    final = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.error(f"[{candidates[i]['id']}] EXCEPTION: {r}")
            final.append({
                "id": candidates[i]["id"],
                "name": candidates[i].get("name", ""),
                "connect": False,
                "error": str(r),
            })
        else:
            final.append(r)

    # Summary
    ok = [r for r in final if r.get("connect") and r.get("tools")]
    conn_only = [r for r in final if r.get("connect") and not r.get("tools")]
    fail = [r for r in final if not r.get("connect")]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(ok)} working / {len(conn_only)} conn-only / {len(fail)} failed / {len(final)} total")
    print(f"{'='*80}")

    for r in sorted(final, key=lambda x: (not x.get("connect"), not x.get("tools"), x.get("id", ""))):
        connected = r.get("connect", False)
        has_tools = bool(r.get("tools"))
        n_tools = len(r.get("tools", []))
        status = "OK  " if has_tools else ("CONN" if connected else "FAIL")
        err = (r.get("error") or "")[:50]
        t = r.get("time_s", 0)
        uses = r.get("use_count", 0)
        print(f"  [{status}] {r['id']:55s} tools={n_tools:3d}  {t:5.1f}s  uses={uses:7d}  {err}")

    # Save full results
    out_dir = ROOT / "data" / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "smithery_test_results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nFull results: {results_path}")

    # Save pool-compatible format for working servers
    pool_entries = []
    for r in final:
        if r.get("connect") and r.get("tools"):
            pool_entries.append({
                "id": r["id"],
                "name": r.get("name", r["id"]),
                "command": r.get("command", "npx"),
                "args": r.get("args", []),
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "source": "smithery",
                "sources": r.get("sources", ["smithery"]),
                "use_count": r.get("use_count", 0),
                "stars": r.get("stars"),
                "status": "probe_ok",
                "tools": r.get("tools", []),
                "tool_count": len(r.get("tools", [])),
            })

    pool_path = out_dir / "smithery_verified.json"
    with open(pool_path, "w") as f:
        json.dump(pool_entries, f, indent=2)
    print(f"Pool-compatible: {len(pool_entries)} servers → {pool_path}")

    total_tools = sum(len(r.get("tools", [])) for r in final if r.get("tools"))
    print(f"Total tools across working servers: {total_tools}")


if __name__ == "__main__":
    asyncio.run(main())
