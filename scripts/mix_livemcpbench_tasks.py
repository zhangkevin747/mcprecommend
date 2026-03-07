"""Mix LiveMCPBench tasks with our benchmark tasks.

Converts LiveMCPBench tasks to our format and produces a combined task file.
Filters for tasks that are viable with MCP tool-use (skips file-dependent tasks).
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# LiveMCPBench category → our category mapping
CATEGORY_MAP = {
    "Office": "office",
    "Lifestyle": "lifestyle",
    "Leisure": "leisure",
    "Finance": "finance",
    "Travel": "travel",
    "Shopping": "shopping",
}


def load_livemcpbench_tasks() -> list[dict]:
    """Load and convert LiveMCPBench tasks to our format."""
    with open(ROOT / "data" / "raw" / "livemcpbench_tasks.json") as f:
        raw = json.load(f)

    converted = []
    skipped = 0
    for t in raw:
        question = t.get("Question", "")
        file_name = t.get("file_name", "")
        category = CATEGORY_MAP.get(t.get("category", ""), "other")
        metadata = t.get("Annotator Metadata", {})
        num_tools = int(metadata.get("Number of tools", "99"))

        # Skip tasks requiring local files (they reference /root/... paths)
        if file_name and "/root/" in file_name:
            skipped += 1
            continue

        # Skip tasks that reference local file creation/manipulation
        # (these need filesystem + specific file setup)
        answers = t.get("answers", "")
        if answers and answers.startswith("file "):
            skipped += 1
            continue

        converted.append({
            "uuid": t.get("task_id", str(uuid.uuid4())),
            "category": category,
            "call_type": "single" if num_tools <= 2 else "multi",
            "source": "livemcpbench",
            "query": question,
            "tools_annotated": metadata.get("Tools", ""),
            "num_tools_annotated": num_tools,
            "num_steps_annotated": int(metadata.get("Number of steps", "0")),
        })

    print(f"LiveMCPBench: {len(converted)} usable / {skipped} skipped (file-dependent) / {len(raw)} total")
    return converted


def load_our_tasks() -> list[dict]:
    """Load our benchmark tasks from all domains."""
    task_files = {
        "search": ROOT / "data" / "search" / "search_0725_single_v2.json",
        "browser": ROOT / "data" / "browser" / "browser_0724_single_v3.json",
        "finance": ROOT / "data" / "finance" / "finance_0724_single_v3.json",
    }

    all_tasks = []
    for domain, path in task_files.items():
        if not path.exists():
            print(f"  Skipping {domain}: {path} not found")
            continue
        with open(path) as f:
            tasks = json.load(f)
        for t in tasks:
            t["source"] = "mcptoolbench"
            # Normalize: ensure query field exists
            if "query" not in t:
                t["query"] = t.get("question", t.get("instruction", ""))
        all_tasks.extend(tasks)
        print(f"  {domain}: {len(tasks)} tasks")

    print(f"Our benchmark: {len(all_tasks)} tasks total")
    return all_tasks


def main():
    print("Loading tasks...\n")

    our_tasks = load_our_tasks()
    print()
    lmcb_tasks = load_livemcpbench_tasks()

    # Combined
    combined = our_tasks + lmcb_tasks
    print(f"\nCombined: {len(combined)} tasks")

    # Stats
    from collections import Counter
    sources = Counter(t.get("source") for t in combined)
    categories = Counter(t.get("category") for t in combined)
    print(f"By source: {dict(sources)}")
    print(f"By category: {dict(categories)}")

    # Save
    out = ROOT / "data" / "tasks_combined.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Also save just the LiveMCPBench tasks separately
    lmcb_out = ROOT / "data" / "tasks_livemcpbench.json"
    with open(lmcb_out, "w") as f:
        json.dump(lmcb_tasks, f, indent=2)
    print(f"LiveMCPBench tasks saved to {lmcb_out}")


if __name__ == "__main__":
    main()
