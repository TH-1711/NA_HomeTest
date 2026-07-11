"""
Step 3: daily job. Wraps scrape.py (ingestion) and uploads ONLY the delta to
the OpenAI vector store created by build_assistant.py.

This script never creates a new assistant or vector store -- it expects one
to already exist (run build_assistant.py once, manually, first). It reuses
that vector store's chunking_strategy for every subsequent upload so chunk
sizing stays consistent across the whole corpus, not just today's changed
files.

Delta detection is done by SHA-256 hash of each article's final saved
Markdown (header + body), not by Zendesk's `updated_at`. `updated_at` is a
reasonable secondary signal but isn't guaranteed to change on every edit
Zendesk considers cosmetic, so hashing the actual bytes we're about to
upload is the only check that can't produce a false "unchanged".

State (which vector-store file_id backs which article, and that article's
last-seen hash) is persisted to --state-file between runs. On the very
first run, state.json won't exist yet -- see bootstrap_state() for how we
reconcile with the vector store's *existing* contents (from
build_assistant.py's initial batch upload) instead of blindly re-uploading
everything on day one.

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py --articles-dir ./articles --state-file ./state/state.json \
                    --build-result ./build_result.json
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI

import scrape as scraper  # reuse build_session/fetch_all_articles/save_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optibot-daily")


def load_client() -> OpenAI:
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. See .env.sample.")
    return OpenAI(api_key=api_key)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_scrape(out_dir: Path, limit: int | None = None) -> list[dict]:
    """Full re-scrape using scrape.py's own pipeline (PULL/VALIDATE/SELECT/
    TRANSFORM/SAVE), so the daily job and the one-off manual scrape can never
    drift into two different implementations of "what a clean article looks
    like"."""
    session = scraper.build_session()
    log.info("SCRAPE: fetching article list from Zendesk API...")
    articles = scraper.fetch_all_articles(session, limit=limit)

    manifest, seen_slugs = [], set()
    skipped = failed = 0
    for article in articles:
        try:
            meta = scraper.save_article(article, out_dir, seen_slugs)
            manifest.append(meta)
        except ValueError as exc:
            skipped += 1
            log.warning("SKIP id=%s title=%r: %s", article.get("id"), article.get("title"), exc)
        except Exception as exc:
            failed += 1
            log.error("FAIL id=%s title=%r: %s", article.get("id"), article.get("title"), exc)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("SCRAPE done: saved=%d skipped=%d failed=%d fetched=%d",
              len(manifest), skipped, failed, len(articles))
    return manifest


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def bootstrap_state(client: OpenAI, vector_store_id: str, chunking_strategy: dict,
                     manifest: list[dict], articles_dir: Path) -> dict:
    """First-ever main.py run: state.json doesn't exist. The vector store
    already holds files from build_assistant.py's initial batch upload, but
    a batch upload doesn't hand back a per-file mapping -- so we reconstruct
    one by listing the vector store's files, looking up each underlying
    File object's filename, and matching that back to a manifest slug.

    Every matched article is recorded with its CURRENT content hash as the
    baseline (we assume "just built today" == "matches what's embedded").
    This means day one only uploads articles that genuinely couldn't be
    matched (new since the initial build); it does not re-upload the whole
    corpus just because state.json was missing."""
    log.info("BOOTSTRAP: no previous state.json -- reconciling with existing vector store contents")

    filename_to_file_id: dict[str, str] = {}
    after = None
    while True:
        page = client.vector_stores.files.list(vector_store_id=vector_store_id, after=after, limit=100)
        for vsf in page.data:
            try:
                f = client.files.retrieve(vsf.id)
                filename_to_file_id[f.filename] = vsf.id
            except Exception as exc:
                log.warning("BOOTSTRAP: could not resolve filename for file_id=%s: %s", vsf.id, exc)
        if not getattr(page, "has_more", False):
            break
        after = page.data[-1].id

    articles_state, matched = {}, 0
    for entry in manifest:
        file_id = filename_to_file_id.get(entry["file"])
        if not file_id:
            # Not found in the vector store -- do NOT add it to state here.
            # Leaving it out of prev_articles means diff_manifest() will
            # correctly see it as brand new and upload it. (Adding a
            # placeholder entry with vector_store_file_id=None here was the
            # original bug: diff_manifest matched it by id/hash and called
            # it "skipped", even though it had never actually been
            # uploaded -- silently leaving articles out of the assistant.)
            continue
        matched += 1
        file_path = articles_dir / entry["file"]
        articles_state[str(entry["id"])] = {
            "slug": entry["slug"],
            "file": entry["file"],
            "hash": sha256_file(file_path),
            "vector_store_file_id": file_id,
        }

    log.info("BOOTSTRAP: matched %d/%d articles to existing vector store files",
              matched, len(manifest))
    return {"vector_store_id": vector_store_id, "chunking_strategy": chunking_strategy,
            "articles": articles_state}


def diff_manifest(manifest: list[dict], prev_articles: dict, articles_dir: Path):
    """Compare the freshly scraped manifest against the previous run's
    per-article hash map. Returns (added, updated, skipped, removed_ids)."""
    current_ids = set()
    added, updated, skipped = [], [], []

    for entry in manifest:
        aid = str(entry["id"])
        current_ids.add(aid)
        current_hash = sha256_file(articles_dir / entry["file"])
        entry["_hash"] = current_hash
        prev = prev_articles.get(aid)
        if prev is None:
            added.append(entry)
        elif prev.get("hash") != current_hash:
            updated.append(entry)
        else:
            skipped.append(entry)

    removed_ids = [aid for aid in prev_articles if aid not in current_ids]
    return added, updated, skipped, removed_ids


def upload_one(client: OpenAI, vector_store_id: str, chunking_strategy: dict,
               articles_dir: Path, entry: dict) -> str:
    """Upload one article's .md to OpenAI Files, attach it to the vector
    store, and poll until embedding completes. Returns the new file_id."""
    file_path = articles_dir / entry["file"]
    with open(file_path, "rb") as fh:
        file_obj = client.files.create(file=fh, purpose="assistants")

    vs_file = client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store_id,
        file_id=file_obj.id,
        chunking_strategy=chunking_strategy,
    )
    if vs_file.status != "completed":
        raise RuntimeError(f"vector store file {file_obj.id} ended in status={vs_file.status}")
    return file_obj.id


def remove_one(client: OpenAI, vector_store_id: str, old_file_id: str | None) -> None:
    """Detach + delete the old File object so an updated/removed article
    doesn't leave a stale duplicate sitting in storage and getting retrieved
    alongside its replacement."""
    if not old_file_id:
        return
    try:
        client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=old_file_id)
    except Exception as exc:
        log.warning("REMOVE: vector store detach failed for file_id=%s: %s", old_file_id, exc)
    try:
        client.files.delete(old_file_id)
    except Exception as exc:
        log.warning("REMOVE: file delete failed for file_id=%s: %s", old_file_id, exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles-dir", default="./articles")
    parser.add_argument("--state-file", default="./state/state.json")
    parser.add_argument("--build-result", default="./build_result.json",
                         help="output of build_assistant.py, used only on the very first run")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    articles_dir = Path(args.articles_dir)
    articles_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file)

    client = load_client()

    # 1. Always re-scrape (per the assignment: "Re-scrape").
    manifest = run_scrape(articles_dir, limit=args.limit)

    # 2. Resolve vector_store_id / chunking_strategy: prefer state.json (once
    #    bootstrapped); fall back to build_result.json on the first-ever run.
    state = load_state(state_path)
    if state:
        vector_store_id = state["vector_store_id"]
        chunking_strategy = state["chunking_strategy"]
        prev_articles = state.get("articles", {})
    else:
        # Look in --build-result first; if not there, fall back to
        # ./build_result.json (build_assistant.py's own default output path)
        # so a first-ever run doesn't require a manual copy step.
        build_result_path = Path(args.build_result)
        if not build_result_path.exists():
            fallback_path = Path("./build_result.json")
            if fallback_path.exists():
                log.info("build-result not found at %s, using %s instead", build_result_path, fallback_path)
                build_result_path = fallback_path
            else:
                raise SystemExit(
                    f"No {state_path}, no {args.build_result}, and no {fallback_path} found. "
                    "Run build_assistant.py once manually first to create the assistant + vector store."
                )
        build_result = json.loads(build_result_path.read_text(encoding="utf-8"))
        vector_store_id = build_result["vector_store_id"]
        chunking_strategy = build_result["chunking_strategy"]
        bootstrap = bootstrap_state(client, vector_store_id, chunking_strategy, manifest, articles_dir)
        prev_articles = bootstrap["articles"]

    # 3. Diff against previous state, by content hash.
    added, updated, skipped, removed_ids = diff_manifest(manifest, prev_articles, articles_dir)
    log.info("DELTA: added=%d updated=%d skipped=%d removed=%d",
              len(added), len(updated), len(skipped), len(removed_ids))

    # 4. Apply changes: remove first, then upload added + updated.
    new_articles_state = dict(prev_articles)

    for aid in removed_ids:
        old = prev_articles[aid]
        log.info("REMOVE id=%s slug=%s (no longer present at source)", aid, old.get("slug"))
        remove_one(client, vector_store_id, old.get("vector_store_file_id"))
        del new_articles_state[aid]

    for entry in updated:
        aid = str(entry["id"])
        old = prev_articles.get(aid, {})
        log.info("UPDATE id=%s slug=%s (content hash changed)", aid, entry["slug"])
        remove_one(client, vector_store_id, old.get("vector_store_file_id"))
        new_file_id = upload_one(client, vector_store_id, chunking_strategy, articles_dir, entry)
        new_articles_state[aid] = {"slug": entry["slug"], "file": entry["file"],
                                    "hash": entry["_hash"], "vector_store_file_id": new_file_id}

    for entry in added:
        aid = str(entry["id"])
        log.info("ADD id=%s slug=%s (new article)", aid, entry["slug"])
        new_file_id = upload_one(client, vector_store_id, chunking_strategy, articles_dir, entry)
        new_articles_state[aid] = {"slug": entry["slug"], "file": entry["file"],
                                    "hash": entry["_hash"], "vector_store_file_id": new_file_id}

    for entry in skipped:
        aid = str(entry["id"])
        if aid in new_articles_state:
            new_articles_state[aid]["hash"] = entry["_hash"]

    save_state(state_path, {
        "vector_store_id": vector_store_id,
        "chunking_strategy": chunking_strategy,
        "articles": new_articles_state,
    })

    log.info("DONE. added=%d updated=%d skipped=%d removed=%d total_tracked=%d",
              len(added), len(updated), len(skipped), len(removed_ids), len(new_articles_state))
    print(f"added={len(added)} updated={len(updated)} skipped={len(skipped)} removed={len(removed_ids)}")


if __name__ == "__main__":
    main()