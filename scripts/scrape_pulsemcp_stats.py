"""Scrape estimated visitor counts and popularity rankings from PulseMCP detail pages.

Reads slugs from existing pulsemcp_remote_servers.json (or combined_server_pool.json)
and fetches only the stats, then merges back.
"""
import json
import re
import time
import urllib.request
import socket

socket.setdefaulttimeout(10)

INPUT_FILE = "data/pulsemcp_remote_servers.json"
OUTPUT_FILE = "data/pulsemcp_remote_servers.json"  # overwrite with enriched data
BASE_URL = "https://www.pulsemcp.com/servers"


def parse_visitor_count(text):
    """Parse '9.4m' or '592k' or '1.2b' into an integer."""
    text = text.strip().lower().replace(",", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if text.endswith(suffix):
            return int(float(text[:-1]) * mult)
    try:
        return int(float(text))
    except ValueError:
        return None


def fetch_stats(slug):
    """Fetch visitor count and ranking from a PulseMCP detail page."""
    url = f"{BASE_URL}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode()
    except Exception as e:
        return {"error": str(e)}

    stats = {}

    # Est. Visitors — value is a loose text node after the info-box div
    vis_match = re.search(
        r'Est\.\s*Visitors.*?</div>\s*\n\s*([0-9][0-9.,]*[kmb]?)\s*\(([0-9][0-9.,]*[kmb]?)\s*this\s+week\)',
        html, re.IGNORECASE | re.DOTALL
    )
    if vis_match:
        stats["est_visitors"] = parse_visitor_count(vis_match.group(1))
        stats["est_visitors_weekly"] = parse_visitor_count(vis_match.group(2))
        stats["est_visitors_raw"] = f"{vis_match.group(1)} ({vis_match.group(2)} this week)"

    # Popularity Ranking — same loose text node pattern
    rank_match = re.search(
        r'Popularity\s+Ranking.*?</div>\s*\n\s*#(\d+)\s*\(#(\d+)\s*this\s+week\)',
        html, re.IGNORECASE | re.DOTALL
    )
    if rank_match:
        stats["popularity_rank"] = int(rank_match.group(1))
        stats["popularity_rank_weekly"] = int(rank_match.group(2))

    # GitHub stars (bonus — grab if present)
    stars_match = re.search(r'([\d,.]+[km]?)\s*stars?', html, re.IGNORECASE)
    if stars_match:
        stats["github_stars_raw"] = stars_match.group(1)
        parsed = parse_visitor_count(stars_match.group(1))
        if parsed is not None:
            stats["github_stars"] = parsed

    return stats


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    servers = data["servers"]
    print(f"Fetching stats for {len(servers)} PulseMCP servers...")

    fetched = 0
    with_visitors = 0
    with_rank = 0
    errors = 0

    for i, server in enumerate(servers):
        slug = server.get("slug")
        if not slug:
            continue

        if i > 0 and i % 20 == 0:
            print(f"  Progress: {i}/{len(servers)} (visitors: {with_visitors}, ranks: {with_rank}, errors: {errors})")

        stats = fetch_stats(slug)
        fetched += 1

        if "error" in stats:
            errors += 1
        else:
            if "est_visitors" in stats:
                with_visitors += 1
            if "popularity_rank" in stats:
                with_rank += 1

        # Merge stats into server record
        server.update(stats)
        time.sleep(0.15)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n--- Stats ---")
    print(f"Fetched: {fetched}")
    print(f"With visitor counts: {with_visitors}")
    print(f"With popularity rank: {with_rank}")
    print(f"Errors: {errors}")
    print(f"Saved to {OUTPUT_FILE}")

    # Top 10 by visitors
    ranked = sorted(
        [s for s in servers if s.get("est_visitors")],
        key=lambda s: s["est_visitors"],
        reverse=True
    )
    if ranked:
        print(f"\nTop 10 by est. visitors:")
        for s in ranked[:10]:
            print(f"  {s.get('name', s['slug']):40s} {s['est_visitors_raw']:>30s}  rank #{s.get('popularity_rank', '?')}")


if __name__ == "__main__":
    main()
