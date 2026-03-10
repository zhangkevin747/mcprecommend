"""Probe all servers in the pool and cache their tool schemas.

Connects to each server, fetches tool schemas, and saves results to
data/pool_schema_cache.json. Servers that fail to connect or return
no tools are marked as unmountable.

Usage:
    python3 -m scripts.probe_pool --concurrency 20
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.recsys.pipeline import _try_connect

log = logging.getLogger(__name__)


async def probe_server(server: dict, sem: asyncio.Semaphore) -> dict:
    server_id = server["id"]
    async with sem:
        t0 = time.time()
        try:
            conn, tools = await _try_connect(server)
            if conn:
                await conn.close()
            latency = round(time.time() - t0, 2)
            if tools:
                return {
                    "id": server_id,
                    "mountable": True,
                    "tools": tools,
                    "tool_count": len(tools),
                    "latency_s": latency,
                }
            else:
                return {"id": server_id, "mountable": False, "tools": [], "tool_count": 0, "latency_s": latency}
        except Exception as e:
            return {"id": server_id, "mountable": False, "tools": [], "tool_count": 0,
                    "error": str(e), "latency_s": round(time.time() - t0, 2)}


async def main(concurrency: int, output: str):
    pool_path = ROOT / "data" / "combined_server_pool.json"
    data = json.loads(pool_path.read_text())
    servers = data["servers"]
    log.info(f"Probing {len(servers)} servers at concurrency={concurrency}...")

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    results = await asyncio.gather(*[probe_server(s, sem) for s in servers])

    mountable = [r for r in results if r["mountable"]]
    total_tools = sum(r["tool_count"] for r in mountable)
    elapsed = time.time() - t0

    log.info(f"Done in {elapsed:.1f}s")
    log.info(f"Mountable: {len(mountable)}/{len(servers)} ({len(mountable)/len(servers)*100:.1f}%)")
    log.info(f"Total tools cached: {total_tools}")

    out = {
        "total": len(servers),
        "mountable": len(mountable),
        "probed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "servers": {r["id"]: r for r in results},
    }
    Path(output).write_text(json.dumps(out, indent=2))
    log.info(f"Saved to {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/pool_schema_cache.json")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.output))
