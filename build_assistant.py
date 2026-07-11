"""
Step 2: create the OptiBot Assistant and load the vector store via the
OpenAI API. Run this after scrape.py has populated ./articles with .md files.

Usage:
    export OPENAI_API_KEY=sk-...   (or put it in a .env file, see .env.sample)
    python build_assistant.py --articles-dir ./articles
"""

import argparse
import json
import logging
import os
from pathlib import Path

import tiktoken
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optibot-build")

# Verbatim system prompt required by the assignment -- do not paraphrase this.
SYSTEM_PROMPT = (
    "You are OptiBot, the customer-support bot for OptiSigns.com.\n"
    "\u2022 Tone: helpful, factual, concise.\n"
    "\u2022 Only answer using the uploaded docs.\n"
    "\u2022 Max 5 bullet points; else link to the doc.\n"
    "\u2022 Cite up to 3 \"Article URL:\" lines per reply."
)

MODEL = "gpt-4o-mini"  # cheap + fast, plenty for a doc-grounded support bot

# OpenAI vector store hard limits (as of the current API):
#   max_chunk_size_tokens: 100-4096
#   chunk_overlap_tokens: must not exceed max_chunk_size_tokens / 2
# Default strategy is 800 / 400. We don't trust that blindly -- see
# decide_chunking_strategy() below.
CHUNK_TOKEN_MIN = 100
CHUNK_TOKEN_MAX = 4096
DEFAULT_MAX_CHUNK = 800

# gpt-4o-mini defaults to returning up to 20 chunks per file_search call.
# System prompt caps the reply at 5 bullets and <=3 cited articles -- feeding
# it 20 chunks is overkill and just burns token budget. See
# decide_max_num_results() for the actual reasoning.
FILE_SEARCH_DEFAULT_MAX_RESULTS = 20


def load_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Copy .env.sample to .env, fill in your key, "
            "then `export $(cat .env | xargs)` or use python-dotenv before running."
        )
    return OpenAI(api_key=api_key)


def compute_token_stats(articles_dir: Path) -> dict:
    """Measure the real token-length distribution of the scraped Markdown
    files, using the same tokenizer family OpenAI's embeddings use
    (cl100k_base). This replaces guessing chunk size with an actual number."""
    encoding = tiktoken.get_encoding("cl100k_base")
    paths = sorted(articles_dir.glob("*.md"))
    if not paths:
        raise SystemExit(f"No .md files found in {articles_dir}. Run scrape.py first.")

    lengths = [len(encoding.encode(p.read_text(encoding="utf-8"))) for p in paths]
    lengths.sort()
    n = len(lengths)

    stats = {
        "n_files": n,
        "min": lengths[0],
        "max": lengths[-1],
        "avg": sum(lengths) / n,
        "median": lengths[n // 2],
        # p90: 90% of articles are at or below this length. We size chunks
        # around this instead of the average, because the average is easily
        # skewed low by many short FAQ articles while a handful of long
        # troubleshooting guides need the bigger chunk to stay in one piece.
        "p90": lengths[min(n - 1, int(n * 0.9))],
    }
    log.info(
        "TOKEN STATS n_files=%d avg=%.0f median=%d p90=%d min=%d max=%d",
        stats["n_files"], stats["avg"], stats["median"], stats["p90"], stats["min"], stats["max"],
    )
    return stats


def decide_chunking_strategy(stats: dict) -> dict:
    """Decide max_chunk_size_tokens / chunk_overlap_tokens from the measured
    token distribution instead of trusting OpenAI's 800/400 default blindly.

    Reasoning:
    - Every file starts with its own "Article URL:" header. If a file's full
      body fits inside ONE chunk, that citation line stays attached to
      whatever content gets retrieved from it -- which is what makes the
      system prompt's "cite Article URL: lines" instruction reliable. If the
      file gets split into multiple chunks, only the chunk containing the
      header keeps the citation; chunks from later in the article don't.
    - So the target is: size chunks to comfortably fit p90 of the corpus,
      not the average. Optimizing for "most articles stay in one chunk" is
      more useful here than optimizing for "the typical article fits".
    - Only raise it above the 800-token default if the measured p90 actually
      needs it -- shrinking or inflating chunk size without evidence just
      changes vector count and cost for no retrieval benefit.
    - Overlap is set to a fixed 50% of chunk size (the platform's max ratio),
      cheap insurance against losing context at a chunk boundary for the
      cases that DO get split.
    """
    p90 = stats["p90"]

    if p90 <= DEFAULT_MAX_CHUNK:
        max_chunk = DEFAULT_MAX_CHUNK
        reason = f"p90={p90} tokens already fits inside the {DEFAULT_MAX_CHUNK}-token default; no change needed"
    else:
        # Round up to the next 100 tokens above p90, capped at the API max.
        max_chunk = min(CHUNK_TOKEN_MAX, ((p90 // 100) + 1) * 100)
        reason = (
            f"p90={p90} tokens exceeds the {DEFAULT_MAX_CHUNK}-token default; "
            f"raised max_chunk_size_tokens to {max_chunk} so ~90% of articles "
            f"still fit in a single chunk (their Article URL header stays attached)"
        )

    overlap = max_chunk // 2  # platform maximum ratio

    log.info(
        "CHUNKING DECISION max_chunk_size_tokens=%d chunk_overlap_tokens=%d -- %s",
        max_chunk, overlap, reason,
    )
    return {"type": "static", "static": {"max_chunk_size_tokens": max_chunk, "chunk_overlap_tokens": overlap}}


def decide_max_num_results(stats: dict) -> int:
    """Decide file_search's max_num_results instead of leaving the default 20.

    Reasoning: the system prompt caps replies at 5 bullets and cites at most
    3 Article URL lines, i.e. content from at most ~3 distinct articles per
    answer. If most articles fit in 1 chunk (see decide_chunking_strategy),
    covering 3 articles needs roughly 3-6 chunks depending on how often an
    answer legitimately needs neighbouring context. We pick 6: enough
    headroom for an answer that spans 3 articles at up to 2 chunks each,
    without pulling in the full default of 20 (which just wastes the
    file_search tool's token budget on marginally-relevant chunks and makes
    it harder for the model to stay within 5 bullets)."""
    chosen = 6
    log.info(
        "MAX_NUM_RESULTS DECISION chosen=%d (default=%d) -- reply is capped at 3 cited "
        "articles, ~1 chunk/article at measured p90=%d tokens, +headroom for split articles",
        chosen, FILE_SEARCH_DEFAULT_MAX_RESULTS, stats["p90"],
    )
    return chosen


def create_vector_store(client: OpenAI, name: str):
    vs = client.vector_stores.create(name=name)
    log.info("created vector store id=%s name=%r", vs.id, name)
    return vs


def upload_articles(client: OpenAI, vector_store_id: str, articles_dir: Path, chunking_strategy: dict):
    """Upload every .md file in articles_dir and attach it to the vector store
    in one batch call, using the chunking_strategy decided from real token
    stats. upload_and_poll blocks until OpenAI has finished
    parsing/chunking/embedding every file, so file_counts below is final.

    Note: chunking_strategy must be passed HERE, to upload_and_poll itself --
    setting it only on vector_stores.create() does not reliably apply to
    files added afterwards (known SDK behavior, not just an assumption)."""
    paths = sorted(articles_dir.glob("*.md"))
    if not paths:
        raise SystemExit(f"No .md files found in {articles_dir}. Run scrape.py first.")

    file_streams = [open(p, "rb") for p in paths]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=file_streams,
            chunking_strategy=chunking_strategy,
        )
    finally:
        for f in file_streams:
            f.close()

    log.info("upload batch status=%s file_counts=%s", batch.status, batch.file_counts)
    return batch, len(paths)


def create_assistant(client: OpenAI, vector_store_id: str, max_num_results: int):
    assistant = client.beta.assistants.create(
        name="OptiBot",
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        tools=[{"type": "file_search", "file_search": {"max_num_results": max_num_results}}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
    log.info("created assistant id=%s model=%s max_num_results=%d", assistant.id, MODEL, max_num_results)
    return assistant


def log_embedding_summary(client: OpenAI, vector_store_id: str, n_files_uploaded: int):
    """Log what actually got embedded. Note: the API exposes file-level status
    (completed/failed/in_progress) via file_counts, but does NOT expose an
    exact chunk count endpoint -- that's an internal implementation detail
    OpenAI doesn't surface. We report file counts precisely and note the
    chunking strategy (explained in README) instead of guessing a chunk number."""
    vs = client.vector_stores.retrieve(vector_store_id)
    log.info(
        "vector store %s: %d files uploaded, file_counts=%s, usage_bytes=%s",
        vector_store_id, n_files_uploaded, vs.file_counts, vs.usage_bytes,
    )
    return {
        "vector_store_id": vector_store_id,
        "files_uploaded": n_files_uploaded,
        "file_counts": vs.file_counts.model_dump() if hasattr(vs.file_counts, "model_dump") else str(vs.file_counts),
        "usage_bytes": vs.usage_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles-dir", default="./articles")
    parser.add_argument("--vector-store-name", default="OptiSigns Support Docs")
    parser.add_argument("--out", default="build_result.json", help="where to save ids for later reuse")
    args = parser.parse_args()

    client = load_client()
    articles_dir = Path(args.articles_dir)

    stats = compute_token_stats(articles_dir)
    chunking_strategy = decide_chunking_strategy(stats)
    max_num_results = decide_max_num_results(stats)

    vs = create_vector_store(client, args.vector_store_name)
    batch, n_files = upload_articles(client, vs.id, articles_dir, chunking_strategy)
    summary = log_embedding_summary(client, vs.id, n_files)

    assistant = create_assistant(client, vs.id, max_num_results)

    result = {
        "assistant_id": assistant.id,
        "vector_store_id": vs.id,
        "token_stats": stats,
        "chunking_strategy": chunking_strategy,
        "max_num_results": max_num_results,
        **summary,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Saved assistant_id/vector_store_id to %s -- keep this for the delta-upload job in step 3.", args.out)

    print("\nNext: open https://platform.openai.com/playground/assistants")
    print(f"  select assistant {assistant.id} and ask: \"How do I add a YouTube video?\"")
    print("  screenshot the reply with its Article URL citations.")


if __name__ == "__main__":
    main()