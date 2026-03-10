"""Semantic retrieval: embed query → cosine similarity against MCP index."""

import json
import logging

import numpy as np
import openai

from .config import (
    EMBEDDING_INDEX_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    MCP_INDEX_PATH,
    OPENAI_API_KEY,
)

log = logging.getLogger(__name__)

_client = None
_embeddings = None
_embedding_index = None
_server_index = None


def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _load_embeddings():
    global _embeddings, _embedding_index
    if _embeddings is None:
        _embeddings = np.load(EMBEDDINGS_PATH)
        with open(EMBEDDING_INDEX_PATH) as f:
            _embedding_index = json.load(f)
    return _embeddings, _embedding_index


def _load_server_index():
    global _server_index
    if _server_index is None:
        with open(MCP_INDEX_PATH) as f:
            data = json.load(f)
        _server_index = {s["id"]: s for s in data["servers"]}
    return _server_index


def embed_query(text: str) -> np.ndarray:
    """Embed a query string using OpenAI text-embedding-3-small."""
    client = _get_client()
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def retrieve_from_pool(query: str, pool: list[dict], top_n: int = 100) -> list[dict]:
    """Retrieve top N servers from a pre-defined pool by cosine similarity.

    For pool servers in the embedding index, uses pre-computed embeddings.
    For others (e.g. benchmark servers), computes embeddings from tool descriptions.
    Returns pool dicts enriched with similarity score and index metadata.
    """
    query_emb = embed_query(query)
    embeddings, emb_index = _load_embeddings()

    # Build ID → embedding index position lookup
    id_to_idx = {}
    for i, entry in enumerate(emb_index):
        id_to_idx[entry["id"]] = i

    pool_embs = []
    pool_entries = []
    missing = []

    for server in pool:
        sid = server["id"]
        if sid in id_to_idx:
            pool_embs.append(embeddings[id_to_idx[sid]])
            pool_entries.append(server)
        else:
            missing.append(server)

    # For servers not in the index, embed their tool descriptions
    if missing:
        client = _get_client()
        texts = []
        for s in missing:
            tool_descs = " ".join(
                t.get("description", t.get("name", ""))[:200]
                for t in s.get("tools", [])
            )
            text = f"{s.get('name', s['id'])}. {tool_descs}"[:500]
            texts.append(text)

        resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        for item, server in zip(sorted(resp.data, key=lambda x: x.index), missing):
            pool_embs.append(np.array(item.embedding, dtype=np.float32))
            pool_entries.append(server)

    pool_emb_matrix = np.array(pool_embs)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    emb_norms = np.linalg.norm(pool_emb_matrix, axis=1, keepdims=True) + 1e-9
    similarities = (pool_emb_matrix / emb_norms) @ query_norm

    top_indices = np.argsort(similarities)[::-1][:top_n]
    server_index = _load_server_index()

    results = []
    for idx in top_indices:
        server = pool_entries[idx]
        sid = server["id"]
        full_meta = server_index.get(sid, {})
        results.append({
            **server,  # keeps command, args, status, tools from pool
            "name": server.get("name", full_meta.get("name", sid)),
            "description": full_meta.get("description", ""),
            "stars": full_meta.get("stars"),
            "category": full_meta.get("category"),
            "sources": full_meta.get("sources", []),
            "use_count": full_meta.get("use_count", 0),
            "is_deployed": full_meta.get("is_deployed", False),
            "similarity": float(similarities[idx]),
        })

    return results


def precompute_pool_embeddings(pool: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Precompute embeddings for all pool servers. Call once at batch startup.

    Uses pre-computed index where available, embeds the rest via one API call.
    Returns (L2-normalized matrix, aligned pool entries).
    """
    embeddings, emb_index = _load_embeddings()
    id_to_idx = {entry["id"]: i for i, entry in enumerate(emb_index)}

    pool_embs = []
    pool_entries = []
    missing = []

    for server in pool:
        sid = server["id"]
        if sid in id_to_idx:
            pool_embs.append(embeddings[id_to_idx[sid]])
            pool_entries.append(server)
        else:
            missing.append(server)

    if missing:
        client = _get_client()
        texts = []
        for s in missing:
            tool_descs = " ".join(
                t.get("description", t.get("name", ""))[:200]
                for t in s.get("tools", [])
            )
            texts.append(f"{s.get('name', s['id'])}. {tool_descs}"[:500])

        # Batch in chunks of 2048 (API limit)
        for i in range(0, len(texts), 2048):
            chunk = texts[i:i + 2048]
            chunk_servers = missing[i:i + 2048]
            resp = client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
            for item, server in zip(sorted(resp.data, key=lambda x: x.index), chunk_servers):
                pool_embs.append(np.array(item.embedding, dtype=np.float32))
                pool_entries.append(server)

    matrix = np.array(pool_embs)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms

    log.info(f"Pool embeddings: {matrix.shape[0]} servers "
             f"({matrix.shape[0] - len(missing)} cached, {len(missing)} embedded)")
    return matrix, pool_entries


def precompute_query_embeddings(tasks: list[dict]) -> dict[str, np.ndarray]:
    """Precompute L2-normalized embeddings for all task queries.

    Returns {task_uuid: normalized_embedding}. One API call for all tasks.
    """
    client = _get_client()
    # Truncate long queries to avoid exceeding embedding model token limit
    queries = [t.get("query", t.get("question", ""))[:2000] for t in tasks]
    uuids = [t.get("uuid") or t.get("task_id", str(i)) for i, t in enumerate(tasks)]

    result = {}
    for i in range(0, len(queries), 256):
        chunk_q = queries[i:i + 2048]
        chunk_u = uuids[i:i + 2048]
        resp = client.embeddings.create(input=chunk_q, model=EMBEDDING_MODEL)
        for item, uid in zip(sorted(resp.data, key=lambda x: x.index), chunk_u):
            emb = np.array(item.embedding, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            result[uid] = emb

    log.info(f"Precomputed {len(result)} query embeddings")
    return result


def retrieve_from_pool_fast(
    query_emb: np.ndarray,
    pool_emb_matrix: np.ndarray,
    pool_entries: list[dict],
    top_n: int = 100,
) -> list[dict]:
    """Fast retrieval using precomputed embeddings. No API calls.

    Both query_emb and pool_emb_matrix must be L2-normalized.
    """
    server_index = _load_server_index()
    similarities = pool_emb_matrix @ query_emb
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        server = pool_entries[idx]
        sid = server["id"]
        full_meta = server_index.get(sid, {})
        results.append({
            **server,
            "name": server.get("name", full_meta.get("name", sid)),
            "description": server.get("description", full_meta.get("description", "")),
            "stars": server.get("stars", full_meta.get("stars")),
            "category": server.get("category", full_meta.get("category")),
            "sources": server.get("sources", full_meta.get("sources", [])),
            "use_count": server.get("use_count", full_meta.get("use_count", 0)),
            "is_deployed": server.get("is_deployed", full_meta.get("is_deployed", False)),
            "similarity": float(similarities[idx]),
        })
    return results


def retrieve(query: str, top_n: int = 100) -> list[dict]:
    """Retrieve top N MCP servers by cosine similarity to query.

    Returns list of dicts with server metadata + similarity score.
    """
    query_emb = embed_query(query)
    embeddings, emb_index = _load_embeddings()
    server_index = _load_server_index()

    # Cosine similarity (embeddings are already L2-normalized by OpenAI)
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    similarities = (embeddings / emb_norms) @ query_norm

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for idx in top_indices:
        entry = emb_index[idx]
        server_id = entry["id"]
        server = server_index.get(server_id, {})
        results.append({
            "id": server_id,
            "name": entry.get("name", server.get("name", server_id)),
            "description": server.get("description", ""),
            "tools": server.get("tools", []),
            "tool_count": server.get("tool_count", 0),
            "stars": server.get("stars"),
            "category": server.get("category"),
            "sources": server.get("sources", []),
            "use_count": server.get("use_count", 0),
            "repo_url": server.get("repo_url", ""),
            "is_deployed": server.get("is_deployed", False),
            "similarity": float(similarities[idx]),
        })

    return results
