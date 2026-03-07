"""
Build unified MCP server index from scraped registry data.

Steps:
1. Load raw data from each registry
2. Normalize to common schema
3. Deduplicate by repo URL / (owner, name)
4. Compute embeddings
5. Generate summary statistics
"""

import json
import os
import re
import sys
from collections import Counter
from urllib.parse import urlparse

# Paths
RAW_DIR = "data/raw"
OUTPUT_DIR = "data"
SMITHERY_FILE = os.path.join(RAW_DIR, "smithery_servers.json")
SMITHERY_DETAILS_FILE = os.path.join(RAW_DIR, "smithery_details.json")
OFFICIAL_FILE = os.path.join(RAW_DIR, "official_registry_servers.json")
GLAMA_FILE = os.path.join(RAW_DIR, "glama_servers.json")

INDEX_FILE = os.path.join(OUTPUT_DIR, "mcp_server_index.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "index_stats.json")
EMBEDDING_INDEX_FILE = os.path.join(OUTPUT_DIR, "embedding_index.json")


def normalize_repo_url(url: str) -> str:
    """Normalize GitHub/GitLab repo URLs for dedup."""
    if not url:
        return ""
    url = url.strip().rstrip("/")
    url = re.sub(r"\.git$", "", url)
    url = re.sub(r"#.*$", "", url)
    # Normalize to lowercase for comparison
    parsed = urlparse(url)
    if parsed.netloc in ("github.com", "gitlab.com"):
        # Keep only owner/repo (first two path components)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            return f"https://{parsed.netloc}/{parts[0]}/{parts[1]}".lower()
    return url.lower()


def load_smithery() -> list:
    """Load and normalize Smithery servers."""
    if not os.path.exists(SMITHERY_FILE):
        print("  Smithery file not found, skipping")
        return []

    with open(SMITHERY_FILE) as f:
        data = json.load(f)

    # Load details if available
    details = {}
    if os.path.exists(SMITHERY_DETAILS_FILE):
        with open(SMITHERY_DETAILS_FILE) as f:
            details = json.load(f)
        print(f"  Loaded tool details for {len(details)} Smithery servers")

    servers = []
    for s in data["servers"]:
        qn = s.get("qualifiedName", "")
        detail = details.get(qn, {})

        # Try to extract repo URL from homepage or qualified name
        repo_url = ""
        homepage = s.get("homepage", "")
        if "github.com" in homepage:
            repo_url = homepage

        # Smithery qualifiedName is often "owner/repo" format
        parts = qn.split("/")
        owner = parts[0] if parts else s.get("namespace", "")
        name = parts[1] if len(parts) > 1 else s.get("displayName", qn)

        tools = detail.get("tools", [])
        tool_names = [t["name"] if isinstance(t, dict) else t for t in tools]

        servers.append({
            "id": qn,
            "name": s.get("displayName", qn),
            "owner": owner,
            "repo_url": repo_url,
            "description": s.get("description", ""),
            "tools": tool_names,
            "tool_count": detail.get("tool_count", len(tool_names)),
            "stars": None,
            "category": None,
            "requires_api_key": None,
            "last_updated": s.get("createdAt"),
            "sources": ["smithery"],
            "use_count": s.get("useCount"),
            "verified": s.get("verified", False),
            "is_deployed": s.get("isDeployed", False),
        })

    print(f"  Loaded {len(servers)} Smithery servers")
    return servers


def load_official_registry() -> list:
    """Load and normalize official MCP registry servers."""
    if not os.path.exists(OFFICIAL_FILE):
        print("  Official registry file not found, skipping")
        return []

    with open(OFFICIAL_FILE) as f:
        data = json.load(f)

    servers = []
    for entry in data["servers"]:
        s = entry.get("server", entry)
        meta = entry.get("_meta", {})
        official_meta = meta.get("io.modelcontextprotocol.registry/official", {})

        repo = s.get("repository", {})
        repo_url = repo.get("url", "")

        # Extract owner from repo URL or name
        name_parts = s.get("name", "").split("/")
        owner = name_parts[0] if len(name_parts) > 1 else ""
        short_name = name_parts[-1] if name_parts else s.get("name", "")

        servers.append({
            "id": s.get("name", ""),
            "name": s.get("title", short_name),
            "owner": owner,
            "repo_url": repo_url,
            "description": s.get("description", ""),
            "tools": [],  # Official registry doesn't list tools in listing
            "tool_count": 0,
            "stars": None,
            "category": None,
            "requires_api_key": None,
            "last_updated": official_meta.get("updatedAt"),
            "sources": ["official"],
            "version": s.get("version"),
            "is_latest": official_meta.get("isLatest", False),
        })

    print(f"  Loaded {len(servers)} official registry servers")
    return servers


def load_glama() -> list:
    """Load and normalize Glama servers."""
    if not os.path.exists(GLAMA_FILE):
        print("  Glama file not found, skipping")
        return []

    with open(GLAMA_FILE) as f:
        data = json.load(f)

    servers = []
    for s in data.get("servers", []):
        namespace = s.get("namespace", "")
        name = s.get("name", s.get("slug", ""))
        repo = s.get("repository", {})
        repo_url = repo.get("url", "") if isinstance(repo, dict) else ""

        # Extract tool names from tools list (Glama includes tool objects)
        tools_raw = s.get("tools", [])
        tool_names = []
        for t in tools_raw:
            if isinstance(t, dict):
                tool_names.append(t.get("name", ""))
            elif isinstance(t, str):
                tool_names.append(t)

        # Build categories from attributes
        attrs = s.get("attributes", [])
        category = None
        if attrs:
            cats = [a.get("value", a) if isinstance(a, dict) else a for a in attrs]
            category = cats[0] if cats else None

        servers.append({
            "id": f"{namespace}/{name}" if namespace else name,
            "name": name,
            "owner": namespace,
            "repo_url": repo_url,
            "description": s.get("description", ""),
            "tools": tool_names,
            "tool_count": len(tool_names),
            "stars": None,
            "category": category,
            "requires_api_key": None,
            "last_updated": None,
            "sources": ["glama"],
        })

    print(f"  Loaded {len(servers)} Glama servers")
    return servers


def extract_github_owner_repo(server: dict) -> str:
    """Extract normalized owner/repo from various server formats."""
    # From repo URL (most reliable)
    repo_url = server.get("repo_url", "")
    if "github.com" in repo_url:
        parts = repo_url.split("github.com/")[-1].strip("/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}".lower().rstrip(".git")

    # From official registry reverse-DNS name: io.github.owner/repo
    server_id = server.get("id", "")
    if server_id.startswith("io.github."):
        remainder = server_id[len("io.github."):]
        parts = remainder.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}".lower()
        elif "." in remainder:
            dot_parts = remainder.split(".")
            if len(dot_parts) >= 2:
                return f"{dot_parts[0]}/{dot_parts[1]}".lower()

    # From Smithery qualifiedName (often matches GitHub owner/repo)
    source = server.get("sources", [])
    if "smithery" in source and "/" in server_id:
        # Smithery qualified names like "owner/repo-name" often match GitHub
        return server_id.lower()

    return ""


def deduplicate(all_servers: list) -> list:
    """Deduplicate servers by GitHub owner/repo, repo URL, then by (owner, name)."""
    # First pass: group by GitHub owner/repo (works across registries)
    by_github = {}
    no_github = []

    for s in all_servers:
        gh_key = extract_github_owner_repo(s)
        if gh_key:
            if gh_key not in by_github:
                by_github[gh_key] = []
            by_github[gh_key].append(s)
        else:
            no_github.append(s)

    # Second pass: for non-GitHub servers, group by normalized repo URL
    by_repo = {}
    no_repo = []

    for s in no_github:
        repo_key = normalize_repo_url(s.get("repo_url", ""))
        if repo_key:
            if repo_key not in by_repo:
                by_repo[repo_key] = []
            by_repo[repo_key].append(s)
        else:
            no_repo.append(s)

    # Third pass: group remaining by (owner, name-slug)
    by_name = {}
    for s in no_repo:
        owner = (s.get("owner") or "").lower().strip()
        name = (s.get("name") or s.get("id") or "").lower().strip()
        # Normalize name: remove common suffixes
        name = re.sub(r"[-_]mcp[-_]?server$", "", name)
        name = re.sub(r"[-_]mcp$", "", name)
        key = (owner, name) if owner else ("", name)
        if key not in by_name:
            by_name[key] = []
        by_name[key].append(s)

    # Merge duplicates
    merged = []

    def merge_group(group: list) -> dict:
        """Merge a group of duplicate server entries."""
        base = group[0].copy()
        for other in group[1:]:
            # Merge sources
            base["sources"] = list(set(base.get("sources", []) + other.get("sources", [])))
            # Take non-empty fields
            if not base.get("description") and other.get("description"):
                base["description"] = other["description"]
            if not base.get("repo_url") and other.get("repo_url"):
                base["repo_url"] = other["repo_url"]
            if not base.get("stars") and other.get("stars"):
                base["stars"] = other["stars"]
            if not base.get("category") and other.get("category"):
                base["category"] = other["category"]
            # Merge tools (union)
            existing_tools = set(base.get("tools", []))
            for t in other.get("tools", []):
                if t not in existing_tools:
                    base["tools"].append(t)
                    existing_tools.add(t)
            base["tool_count"] = len(base.get("tools", []))
            # Keep highest use_count
            if other.get("use_count") and (not base.get("use_count") or other["use_count"] > base["use_count"]):
                base["use_count"] = other["use_count"]
        return base

    for group in by_github.values():
        merged.append(merge_group(group))
    for group in by_repo.values():
        merged.append(merge_group(group))
    for group in by_name.values():
        merged.append(merge_group(group))

    return merged


def compute_stats(servers: list, raw_counts: dict) -> dict:
    """Compute summary statistics."""
    source_counts = Counter()
    source_only = Counter()
    tool_counts = []
    categories = Counter()
    has_repo = 0
    has_tools = 0

    for s in servers:
        sources = s.get("sources", [])
        for src in sources:
            source_counts[src] += 1
        if len(sources) == 1:
            source_only[sources[0]] += 1

        tc = s.get("tool_count", 0) or len(s.get("tools", []))
        tool_counts.append(tc)
        if tc > 0:
            has_tools += 1

        if s.get("repo_url"):
            has_repo += 1

        cat = s.get("category")
        if cat:
            categories[cat] += 1

    # Overlapping servers
    multi_source = sum(1 for s in servers if len(s.get("sources", [])) > 1)

    stats = {
        "total_unique_servers": len(servers),
        "raw_counts_per_source": raw_counts,
        "total_raw_listings": sum(raw_counts.values()),
        "duplicates_removed": sum(raw_counts.values()) - len(servers),
        "source_distribution": dict(source_counts),
        "source_only_counts": dict(source_only),
        "multi_source_servers": multi_source,
        "servers_with_repo": has_repo,
        "servers_with_tools": has_tools,
        "tool_count_distribution": {
            "min": min(tool_counts) if tool_counts else 0,
            "max": max(tool_counts) if tool_counts else 0,
            "mean": round(sum(tool_counts) / len(tool_counts), 2) if tool_counts else 0,
            "median": sorted(tool_counts)[len(tool_counts) // 2] if tool_counts else 0,
        },
        "top_20_categories": dict(categories.most_common(20)),
    }
    return stats


def main():
    print("Loading raw data...")
    smithery = load_smithery()
    official = load_official_registry()
    glama = load_glama()

    raw_counts = {
        "smithery": len(smithery),
        "official": len(official),
        "glama": len(glama),
    }

    all_servers = smithery + official + glama
    print(f"\nTotal raw listings: {len(all_servers)}")

    # For official registry, filter to latest versions only
    official_latest = []
    official_names_seen = set()
    for s in official:
        if s.get("is_latest", True):
            if s["id"] not in official_names_seen:
                official_latest.append(s)
                official_names_seen.add(s["id"])
    print(f"Official registry: {len(official)} total -> {len(official_latest)} unique (latest only)")

    all_for_dedup = smithery + official_latest + glama
    raw_counts["official_latest"] = len(official_latest)

    print("\nDeduplicating...")
    merged = deduplicate(all_for_dedup)
    print(f"After dedup: {len(merged)} unique servers")
    print(f"Duplicates removed: {len(all_for_dedup) - len(merged)}")

    # Sort by use_count (if available) then name
    merged.sort(key=lambda s: (-(s.get("use_count") or 0), s.get("name", "").lower()))

    print("\nComputing statistics...")
    stats = compute_stats(merged, raw_counts)

    # Save
    print("\nSaving...")
    with open(INDEX_FILE, "w") as f:
        json.dump({"servers": merged, "total": len(merged)}, f, indent=2)
    print(f"  Index: {INDEX_FILE} ({len(merged)} servers)")

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {STATS_FILE}")

    # Print stats
    print("\n=== Summary Statistics ===")
    print(f"Total unique servers: {stats['total_unique_servers']}")
    print(f"Total raw listings: {stats['total_raw_listings']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Source distribution: {stats['source_distribution']}")
    print(f"Multi-source servers: {stats['multi_source_servers']}")
    print(f"Servers with repo URL: {stats['servers_with_repo']}")
    print(f"Servers with tools: {stats['servers_with_tools']}")
    print(f"Tool counts: {stats['tool_count_distribution']}")

    # Save embedding index (list of server IDs for later embedding)
    embedding_index = [{"idx": i, "id": s["id"], "name": s.get("name", "")} for i, s in enumerate(merged)]
    with open(EMBEDDING_INDEX_FILE, "w") as f:
        json.dump(embedding_index, f, indent=2)
    print(f"  Embedding index: {EMBEDDING_INDEX_FILE}")


if __name__ == "__main__":
    main()
