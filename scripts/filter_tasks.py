"""
Filter and augment tasks:
1. DROP tasks requiring personal state (my inbox, my calendar, my repo, etc.)
2. AUGMENT tasks that need a concrete artifact with paths to data/artifacts/
3. KEEP self-contained tasks as-is
"""

import json
import os
import time
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_DIR = Path("data")
TASKS_FILE = DATA_DIR / "tasks.json"
TASKS_FILTERED_FILE = DATA_DIR / "tasks_filtered.json"

ARTIFACTS = {
    "pdf": "data/artifacts/attention_is_all_you_need.pdf",
    "image": "data/artifacts/sample_image.png",
    "csv": "data/artifacts/sample_data.csv",
    "video": "data/artifacts/sample_video.mp4",
    "audio": "data/artifacts/sample_audio.mp3",
    "youtube": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
}

CLASSIFY_MODEL = "gpt-4o-mini"
BATCH_SIZE = 25


def classify_batch(tasks):
    """Classify a batch of tasks as KEEP, DROP, or AUGMENT."""
    task_list = "\n".join(
        f'{i}. {t["query"]}' for i, t in enumerate(tasks)
    )

    prompt = f"""You are classifying tasks for an AI tool recommender evaluation.

Available artifacts for augmentation:
- pdf: A research paper PDF (Attention Is All You Need)
- csv: A sales data CSV (products, regions, revenue, ratings over 6 months)
- image: A PNG image (3D rendered teacup)
- video: A short MP4 video clip (10 seconds, Big Buck Bunny)
- audio: A short MP3 audio clip (horse neighing sound)
- youtube: A YouTube video URL

For each task below, classify it as one of:
- KEEP: Requires a tool AND can be completed with just a tool call and public information (search, weather, stock prices, currency conversion, live data, public APIs, etc.)
- DROP_PERSONAL: Requires personal/authenticated state that won't exist (my inbox, my files, my calendar, my repo, my server, my account, my team, my contacts, my database, specific user data, etc.)
- DROP_PARAMETRIC: Can be answered from general LLM knowledge without any tool. Tasks like translation, writing, generating greetings, drafting text, explaining concepts, creative writing, etc. If an LLM could answer it well without calling any external tool, drop it.
- AUGMENT:<artifact_type>: Needs a concrete file/media input to work. Replace the vague reference with the specific artifact. Only use if the task fundamentally requires a file input (PDF analysis, image processing, audio transcription, video analysis, CSV/data analysis). The artifact_type must be one of: pdf, csv, image, video, audio, youtube.

Tasks:
{task_list}

Return a JSON object mapping task index (as string) to classification. Example:
{{"0": "KEEP", "1": "DROP_PERSONAL", "2": "AUGMENT:pdf", "3": "DROP_PARAMETRIC"}}

Be strict about DROP_PERSONAL — if the task says "my", "our", or assumes the user has specific pre-existing data/accounts/files/projects, drop it. But "my" referring to the agent's own action (like "find me...") is fine.

Be strict about DROP_PARAMETRIC — if the task is about generating text, translating, writing, explaining, or anything an LLM already knows, drop it. But tasks requiring LIVE or REAL-TIME data (current weather, stock prices, exchange rates, current time, live transit schedules) should be KEPT — those genuinely need a tool.

Be conservative about AUGMENT — only use it when the task genuinely needs a file input, not just because it mentions a document conceptually."""

    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        print("  Failed to parse classification response")
        return {}


def rewrite_task_with_artifact(task, artifact_type):
    """Rewrite a task to reference a specific artifact."""
    artifact_path = ARTIFACTS[artifact_type]

    artifact_descriptions = {
        "pdf": f"the research paper at '{artifact_path}' (Attention Is All You Need by Vaswani et al.)",
        "csv": f"the sales data CSV at '{artifact_path}' (product sales by region over 6 months)",
        "image": f"the image at '{artifact_path}' (a 3D rendered teacup)",
        "video": f"the video at '{artifact_path}' (a short Big Buck Bunny clip)",
        "audio": f"the audio file at '{artifact_path}' (a short horse sound clip)",
        "youtube": f"the YouTube video at '{artifact_path}'",
    }

    prompt = f"""Rewrite this task to reference a specific artifact instead of a vague/personal file.

Original task: {task["query"]}
Artifact: {artifact_descriptions[artifact_type]}

Rewrite the task so it references this specific artifact naturally. Keep the same intent and complexity.
Return ONLY the rewritten task as a plain string, no quotes or JSON."""

    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip().strip('"')


def main():
    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks")

    # Classify in batches
    classifications = {}
    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i:i + BATCH_SIZE]
        print(f"  Classifying batch {i//BATCH_SIZE + 1}/{(len(tasks) + BATCH_SIZE - 1)//BATCH_SIZE}...")
        result = classify_batch(batch)
        for idx_str, label in result.items():
            global_idx = i + int(idx_str)
            classifications[global_idx] = label
        time.sleep(0.3)

    # Tally
    keep_count = sum(1 for v in classifications.values() if v == "KEEP")
    drop_personal = sum(1 for v in classifications.values() if v == "DROP_PERSONAL" or v == "DROP")
    drop_parametric = sum(1 for v in classifications.values() if v == "DROP_PARAMETRIC")
    augment_count = sum(1 for v in classifications.values() if v.startswith("AUGMENT"))
    unclassified = len(tasks) - len(classifications)

    print(f"\nClassification results:")
    print(f"  KEEP:           {keep_count}")
    print(f"  DROP_PERSONAL:  {drop_personal}")
    print(f"  DROP_PARAMETRIC:{drop_parametric}")
    print(f"  AUGMENT:        {augment_count}")
    print(f"  Unclassified:   {unclassified}")

    # Augment breakdown
    augment_types = {}
    for v in classifications.values():
        if v.startswith("AUGMENT:"):
            atype = v.split(":")[1]
            augment_types[atype] = augment_types.get(atype, 0) + 1
    if augment_types:
        print(f"  Augment breakdown: {augment_types}")

    # Process tasks
    final_tasks = []
    augmented = 0
    for i, task in enumerate(tasks):
        label = classifications.get(i, "KEEP")  # default keep if unclassified

        if label in ("DROP", "DROP_PERSONAL", "DROP_PARAMETRIC"):
            continue
        elif label.startswith("AUGMENT:"):
            artifact_type = label.split(":")[1]
            if artifact_type in ARTIFACTS:
                print(f"  Rewriting task {task['task_id']}: {task['query'][:80]}...")
                new_query = rewrite_task_with_artifact(task, artifact_type)
                task["query_original"] = task["query"]
                task["query"] = new_query
                task["artifact_type"] = artifact_type
                task["artifact_path"] = ARTIFACTS[artifact_type]
                augmented += 1
                time.sleep(0.2)
            # If unknown artifact type, keep as-is
        # KEEP or augmented — add to final
        final_tasks.append(task)

    # Re-index task IDs
    for i, t in enumerate(final_tasks):
        t["task_id"] = f"task_{i:04d}"

    with open(TASKS_FILTERED_FILE, "w") as f:
        json.dump(final_tasks, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Final: {len(final_tasks)} tasks saved to {TASKS_FILTERED_FILE}")
    print(f"  Kept as-is:  {len(final_tasks) - augmented}")
    print(f"  Augmented:   {augmented}")
    print(f"  Dropped:     {len(tasks) - len(final_tasks)}")
    print(f"  Clusters:    {len(set(t['cluster_id'] for t in final_tasks))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
