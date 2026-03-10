"""Scrape remote MCP servers from PulseMCP directory.

Only scrapes listing pages (not detail pages) for speed/reliability.
Extracts name, description, and slug from the listing HTML.
"""
import json
import re
import time
import urllib.request
import socket

OUTPUT_FILE = "data/pulsemcp_remote_servers.json"
BASE_URL = "https://www.pulsemcp.com/servers"
FILTER = "other%5B%5D=remote"

# Short timeout to avoid hanging
socket.setdefaulttimeout(10)


def get_page(page_num):
    """Fetch a listing page and extract server info from HTML."""
    url = f"{BASE_URL}?{FILTER}&page={page_num}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
    except Exception as e:
        print(f"  Failed page {page_num}: {e}")
        return [], False

    # Extract server slugs
    slugs = re.findall(r'href="/servers/([^"?]+)"', html)
    seen = set()
    unique_slugs = []
    skip = {"", "new", "submit"}
    for s in slugs:
        if s not in seen and s not in skip and not s.startswith("?"):
            seen.add(s)
            unique_slugs.append(s)

    # Check if there's a next page
    has_next = f"page={page_num + 1}" in html

    return unique_slugs, has_next


def get_server_detail_batch(slugs):
    """Fetch detail pages for a batch of slugs."""
    servers = []
    for slug in slugs:
        url = f"{BASE_URL}/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode()
        except Exception as e:
            # Still add with basic info
            servers.append({"slug": slug, "url": url, "error": str(e)})
            continue

        server = {"slug": slug, "url": url}

        # Title from h1
        title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
        if title_match:
            server["name"] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

        # Description from meta
        desc_match = re.search(r'<meta name="description" content="([^"]*)"', html)
        if desc_match:
            raw = desc_match.group(1).strip()
            # Clean up PulseMCP boilerplate prefix
            raw = re.sub(r'^.*?MCP server[:\s]*', '', raw, count=1, flags=re.IGNORECASE) or raw
            server["description"] = raw

        # GitHub repo
        github_match = re.search(r'href="(https://github\.com/[^"]+)"', html)
        if github_match:
            server["repository"] = github_match.group(1)

        # Remote endpoints (SSE/HTTP)
        sse_matches = re.findall(r'(https?://[^\s"<>\']+(?:sse|/mcp)[^\s"<>\']*)', html, re.IGNORECASE)
        if sse_matches:
            # Filter out obvious non-endpoints
            endpoints = [u for u in set(sse_matches) if 'github.com' not in u and 'pulsemcp.com' not in u]
            if endpoints:
                server["remote_endpoints"] = endpoints

        # npm package
        npm_match = re.search(r'npx\s+(?:-y\s+)?([^\s<"\']+)', html)
        if npm_match:
            server["npm_package"] = npm_match.group(1)

        servers.append(server)
        time.sleep(0.15)

    return servers


def main():
    # Phase 1: Get all slugs from listing pages
    print("Phase 1: Scraping listing pages for server slugs...")
    all_slugs = []
    page = 1
    while True:
        slugs, has_next = get_page(page)
        if not slugs:
            break
        all_slugs.extend(slugs)
        print(f"  Page {page}: {len(slugs)} servers (total: {len(all_slugs)})")
        if not has_next:
            break
        page += 1
        time.sleep(0.2)

    # Dedupe
    seen = set()
    unique_slugs = []
    for s in all_slugs:
        if s not in seen:
            seen.add(s)
            unique_slugs.append(s)

    print(f"\nTotal unique slugs: {len(unique_slugs)}")

    # Phase 2: Fetch detail pages
    print(f"\nPhase 2: Fetching {len(unique_slugs)} detail pages...")
    all_servers = []
    batch_size = 20
    for i in range(0, len(unique_slugs), batch_size):
        batch = unique_slugs[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(unique_slugs) + batch_size - 1)//batch_size} ({i+1}-{min(i+batch_size, len(unique_slugs))})")
        servers = get_server_detail_batch(batch)
        all_servers.extend(servers)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "total": len(all_servers),
            "source": "pulsemcp.com",
            "filter": "remote",
            "servers": all_servers,
        }, f, indent=2)

    # Stats
    with_desc = sum(1 for s in all_servers if s.get("description"))
    with_endpoints = sum(1 for s in all_servers if s.get("remote_endpoints"))
    with_npm = sum(1 for s in all_servers if s.get("npm_package"))
    with_repo = sum(1 for s in all_servers if s.get("repository"))
    errored = sum(1 for s in all_servers if s.get("error"))

    print(f"\n--- Stats ---")
    print(f"Total remote servers: {len(all_servers)}")
    print(f"With description: {with_desc}")
    print(f"With remote endpoints: {with_endpoints}")
    print(f"With npm package: {with_npm}")
    print(f"With GitHub repo: {with_repo}")
    print(f"Errored: {errored}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
