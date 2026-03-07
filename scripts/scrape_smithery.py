"""Scrape all servers from Smithery registry API."""
import json
import time
import urllib.request

OUTPUT_PATH = "data/smithery_servers_all.json"
API_BASE = "https://registry.smithery.ai/servers"
PAGE_SIZE = 50


def scrape_all_servers():
    all_servers = []
    page = 1
    total_pages = None

    while True:
        url = f"{API_BASE}?page={page}&pageSize={PAGE_SIZE}"
        print(f"Fetching page {page}" + (f"/{total_pages}" if total_pages else ""))

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        servers = data.get("servers", [])
        pagination = data.get("pagination", {})
        total_pages = pagination.get("totalPages", "?")

        all_servers.extend(servers)

        if page >= pagination.get("totalPages", page):
            break
        page += 1
        time.sleep(0.2)  # be polite

    print(f"\nTotal servers scraped: {len(all_servers)}")
    return all_servers


def main():
    servers = scrape_all_servers()

    remote_servers = [s for s in servers if s.get("remote")]
    print(f"Remote servers: {len(remote_servers)}")
    print(f"Non-remote servers: {len(servers) - len(remote_servers)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "total": len(servers),
            "remote_count": len(remote_servers),
            "servers": servers,
        }, f, indent=2)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
