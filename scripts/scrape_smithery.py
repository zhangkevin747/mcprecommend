"""Scrape all servers from Smithery registry API with deduplication."""
import json
import time
import urllib.request

OUTPUT_ALL = "data/smithery_servers_all.json"
OUTPUT_REMOTE = "data/smithery_remote_servers.json"
API_BASE = "https://registry.smithery.ai/servers"
PAGE_SIZE = 50


def scrape_all_servers():
    seen_ids = set()
    all_servers = []
    page = 1
    total_pages = None
    dupes_skipped = 0

    while True:
        url = f"{API_BASE}?page={page}&pageSize={PAGE_SIZE}"
        print(f"Fetching page {page}" + (f"/{total_pages}" if total_pages else ""))

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        servers = data.get("servers", [])
        pagination = data.get("pagination", {})
        total_pages = pagination.get("totalPages", "?")

        for s in servers:
            key = s.get("qualifiedName", s.get("id", ""))
            if key and key not in seen_ids:
                seen_ids.add(key)
                all_servers.append(s)
            else:
                dupes_skipped += 1

        if page >= pagination.get("totalPages", page):
            break
        page += 1
        time.sleep(0.2)

    print(f"\nTotal entries from API: {len(seen_ids) + dupes_skipped}")
    print(f"Duplicates skipped: {dupes_skipped}")
    print(f"Unique servers: {len(all_servers)}")
    return all_servers


def main():
    servers = scrape_all_servers()

    with open(OUTPUT_ALL, "w") as f:
        json.dump({
            "total": len(servers),
            "servers": servers,
        }, f, indent=2)
    print(f"Saved all {len(servers)} servers to {OUTPUT_ALL}")

    remote_servers = [s for s in servers if s.get("remote")]
    with open(OUTPUT_REMOTE, "w") as f:
        json.dump({
            "total": len(remote_servers),
            "servers": remote_servers,
        }, f, indent=2)
    print(f"Saved {len(remote_servers)} remote servers to {OUTPUT_REMOTE}")

    # Stats
    non_remote = len(servers) - len(remote_servers)
    verified = sum(1 for s in servers if s.get("verified"))
    deployed = sum(1 for s in remote_servers if s.get("isDeployed"))
    use_counts = sorted([s.get("useCount", 0) for s in remote_servers], reverse=True)
    zero_use = sum(1 for u in use_counts if u == 0)

    print(f"\n--- Stats ---")
    print(f"Total unique: {len(servers)}")
    print(f"Remote: {len(remote_servers)}")
    print(f"Non-remote: {non_remote}")
    print(f"Verified: {verified}")
    print(f"isDeployed: {deployed}")
    if use_counts:
        print(f"Top useCount: {use_counts[0]}")
        print(f"Median useCount: {use_counts[len(use_counts)//2]}")
        print(f"Zero usage: {zero_use} ({zero_use*100//len(remote_servers)}%)")


if __name__ == "__main__":
    main()
