"""Compute embeddings for all MCP servers using OpenAI text-embedding-3-small."""

import json
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system(f"{sys.executable} -m pip install numpy -q")
    import numpy as np

try:
    import openai
except ImportError:
    print("Installing openai...")
    os.system(f"{sys.executable} -m pip install openai -q")
    import openai

from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "data/index/mcp_server_index.json"
EMBEDDINGS_FILE = "data/index/embeddings.npy"
EMBEDDING_INDEX_FILE = "data/index/embedding_index.json"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


def build_text(server: dict) -> str:
    """Build text to embed: name + description + tool names."""
    parts = []
    name = server.get("name", "")
    if name:
        parts.append(name)
    desc = server.get("description", "")
    if desc:
        parts.append(desc)
    tools = server.get("tools", [])
    if tools:
        tool_str = ", ".join(tools[:20])  # Cap at 20 tools
        parts.append(f"Tools: {tool_str}")
    return " | ".join(parts)


def main():
    # Load index
    with open(INDEX_FILE) as f:
        data = json.load(f)
    servers = data["servers"]
    print(f"Loaded {len(servers)} servers")

    # Build texts
    texts = [build_text(s) for s in servers]
    # Filter empty texts
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    print(f"Servers with text to embed: {len(valid)}")

    # Initialize OpenAI client
    client = openai.OpenAI()

    # Embed in batches
    all_embeddings = [None] * len(servers)
    total_tokens = 0

    for batch_start in range(0, len(valid), BATCH_SIZE):
        batch = valid[batch_start:batch_start + BATCH_SIZE]
        batch_texts = [t for _, t in batch]
        batch_indices = [i for i, _ in batch]

        try:
            response = client.embeddings.create(
                model=MODEL,
                input=batch_texts,
            )
            total_tokens += response.usage.total_tokens

            for j, emb_data in enumerate(response.data):
                idx = batch_indices[j]
                all_embeddings[idx] = emb_data.embedding

            done = min(batch_start + BATCH_SIZE, len(valid))
            print(f"  Embedded {done}/{len(valid)} ({total_tokens} tokens)")

        except Exception as e:
            print(f"  Error at batch {batch_start}: {e}")
            time.sleep(5)
            # Retry once
            try:
                response = client.embeddings.create(
                    model=MODEL,
                    input=batch_texts,
                )
                total_tokens += response.usage.total_tokens
                for j, emb_data in enumerate(response.data):
                    idx = batch_indices[j]
                    all_embeddings[idx] = emb_data.embedding
            except Exception as e2:
                print(f"  Retry failed: {e2}")
                # Fill with zeros
                dim = 1536  # text-embedding-3-small dimension
                for idx in batch_indices:
                    all_embeddings[idx] = [0.0] * dim

        time.sleep(0.1)  # Rate limit courtesy

    # Fill any None entries with zeros
    dim = len(all_embeddings[0]) if all_embeddings[0] else 1536
    for i in range(len(all_embeddings)):
        if all_embeddings[i] is None:
            all_embeddings[i] = [0.0] * dim

    # Save
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings_array)
    print(f"\nSaved embeddings: {EMBEDDINGS_FILE} (shape: {embeddings_array.shape})")

    # Save embedding index
    embedding_index = [
        {"idx": i, "id": s["id"], "name": s.get("name", "")}
        for i, s in enumerate(servers)
    ]
    with open(EMBEDDING_INDEX_FILE, "w") as f:
        json.dump(embedding_index, f, indent=2)
    print(f"Saved index: {EMBEDDING_INDEX_FILE}")

    cost = total_tokens / 1_000_000 * 0.02  # $0.02 per 1M tokens
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
