"""
Scrape OptiSigns Help Center articles via the Zendesk Help Center API
and convert each one to a clean Markdown file.

Zendesk exposes a public, unauthenticated REST API for Help Center content:
    GET https://<subdomain>/api/v2/help_center/en-us/articles.json?page=N&per_page=100

The JSON response gives us the article body as raw HTML (just the article
content, no nav/sidebar/footer), so we skip the usual "scrape + strip junk"
problem entirely and go straight to HTML -> Markdown conversion.

Pipeline per article, each stage logged explicitly:
    1. PULL      -- what raw fields Zendesk actually gave us
    2. VALIDATE  -- null/gating checks that decide keep-or-skip
    3. SELECT    -- which fields we keep vs drop vs modify
    4. TRANSFORM -- HTML body -> Markdown
    5. SAVE      -- final field set written to disk + manifest

Usage:
    python scrape.py --out ./articles --limit 40
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from slugify import slugify
from urllib3.util.retry import Retry

BASE_URL = "https://support.optisigns.com/api/v2/help_center/en-us/articles.json"
PER_PAGE = 100
REQUEST_DELAY_SEC = 0.3  # be polite to the API

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("optibot-scraper")

# --- Step 3 (SELECT) field policy -------------------------------------------
# Fields we actually keep in the output Markdown/manifest.
FIELDS_KEPT = ["id", "title", "body", "html_url", "updated_at"]

# Fields we look at only to decide skip/keep, but never write to output.
FIELDS_VALIDATION_ONLY = ["draft", "user_segment_id"]

# Fields we intentionally ignore -- Zendesk operational metadata that adds
# no retrieval value for a support-doc RAG pipeline (vote counts, authoring
# info, position/sort order, comment settings, etc). Listed explicitly here
# so the drop decision is documented, not implicit.
FIELDS_KNOWN_IGNORED = [
    "created_at", "author_id", "comments_disabled", "vote_sum", "vote_count",
    "section_id", "promoted", "position", "label_names", "outdated",
    "outdated_locales", "edited_at", "permission_group_id", "content_tag_ids",
]


def build_session() -> requests.Session:
    """Session with connection reuse + retry/backoff for transient failures
    (429/500/502/503/504). Using urllib3's built-in Retry instead of pulling
    in tenacity as a dependency -- same effect, one less thing to install."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_all_articles(session: requests.Session, limit: int | None = None) -> list[dict]:
    """Paginate through the Zendesk Help Center API and return raw article dicts.
    This is STEP 1 (PULL) at the batch level; per-article raw-field logging
    happens in log_raw_pull() once we're processing each one individually."""
    articles = []
    url = f"{BASE_URL}?per_page={PER_PAGE}"
    logged_schema_sample = False

    while url:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        page_articles = data.get("articles", [])
        articles.extend(page_articles)
        log.info("PULL: fetched page, running total = %d", len(articles))

        # Log the raw field schema once, from the first article we see, so we
        # have a concrete record of "this is what Zendesk actually sent us"
        # rather than trusting the API docs blindly.
        if not logged_schema_sample and page_articles:
            sample = page_articles[0]
            log.info("PULL: raw field schema sample (id=%s) = %s",
                      sample.get("id"), sorted(sample.keys()))
            logged_schema_sample = True

        if limit and len(articles) >= limit:
            articles = articles[:limit]
            break

        url = data.get("next_page")  # Zendesk gives you the full next-page URL directly
        if url:
            time.sleep(REQUEST_DELAY_SEC)

    return articles


def log_raw_pull(article: dict) -> None:
    """STEP 1 -- PULL. Log exactly what raw fields this specific article has
    (some articles omit optional fields, e.g. no label_names), before any
    decision is made about what to do with them."""
    log.debug("PULL id=%s raw_fields=%s", article.get("id"), sorted(article.keys()))


def validate_article(article: dict) -> tuple[bool, str | None]:
    """STEP 2 -- VALIDATE. Null-checks and gating rules, run BEFORE we spend
    any time transforming content. Returns (is_valid, reason_if_invalid)."""
    if article.get("draft"):
        return False, "draft=true (unpublished content, must not train the bot on it)"
    if article.get("user_segment_id"):
        return False, "user_segment_id set (access-restricted content, would leak gated info to a public bot)"
    if not (article.get("title") or "").strip():
        return False, "missing/empty title"
    if not (article.get("body") or "").strip():
        return False, "missing/empty body"
    if not article.get("html_url"):
        return False, "missing html_url (no citation target for the bot)"
    if not article.get("updated_at"):
        return False, "missing updated_at (breaks delta-detection in step 3)"
    return True, None


def select_and_transform_fields(article: dict) -> dict:
    """STEP 3 -- SELECT. Decide, explicitly and logged, which raw fields we
    keep as-is, which we drop, and which we're about to modify. Returns a
    dict containing only the kept raw fields (body still HTML at this point --
    the actual HTML->Markdown modification happens in html_to_markdown)."""
    raw_keys = sorted(article.keys())
    kept = {k: article[k] for k in FIELDS_KEPT if k in article}

    dropped = [k for k in raw_keys if k not in FIELDS_KEPT and k not in FIELDS_VALIDATION_ONLY]
    unexpected = [k for k in dropped if k not in FIELDS_KNOWN_IGNORED]

    log.info(
        "SELECT id=%s kept=%s dropped=%d modified=['title->slug', 'body->markdown']",
        article.get("id"), list(kept.keys()), len(dropped),
    )
    if unexpected:
        # Zendesk added a field we haven't seen/decided on before -- surface it
        # instead of silently dropping, in case it's actually worth keeping.
        log.warning("SELECT id=%s unrecognized fields dropped without review: %s",
                     article.get("id"), unexpected)

    return kept


def html_to_markdown(html: str) -> str:
    """STEP 4 -- TRANSFORM. Convert Zendesk article body HTML to Markdown,
    preserving structure (headings, code blocks, links, tables, lists)."""
    markdown = md(
        html,
        heading_style="ATX",
        bullets="-",
        code_language="",
    )

    # markdownify sometimes leaves raw HTML entities/tags behind (nbsp, stray <br>)
    # when the source HTML has inline styles or empty formatting spans.
    markdown = markdown.replace("&nbsp;", " ")
    markdown = re.sub(r"<br\s*/?>", "\n", markdown)
    markdown = re.sub(r"\*{4,}", "**", markdown)  # collapse stray **** from empty bold spans

    # Collapse 3+ blank lines down to 2, Zendesk HTML tends to be spacing-heavy
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def save_article(article: dict, out_dir: Path, seen_slugs: set) -> dict:
    """Runs the full per-article pipeline (validate -> select -> transform ->
    save) and returns the manifest entry, or raises if the article is invalid
    (caller decides whether that counts as skipped vs failed)."""
    log_raw_pull(article)

    is_valid, reason = validate_article(article)
    if not is_valid:
        raise ValueError(f"validation failed: {reason}")

    kept = select_and_transform_fields(article)
    title = kept["title"]

    # Two different titles can normalize to the same slug (e.g. "How to Login"
    # vs "How-to Login" both -> "how-to-login"), which would silently overwrite
    # a previously written file. Disambiguate on collision using the article id.
    base_slug = slugify(title)[:80] or f"article-{kept['id']}"
    slug = base_slug
    if slug in seen_slugs:
        slug = f"{kept['id']}-{base_slug}"[:100]
    seen_slugs.add(slug)

    filepath = out_dir / f"{slug}.md"

    body_html = kept["body"]
    body_md = html_to_markdown(body_html)
    log.info("TRANSFORM id=%s body html_len=%d -> markdown_len=%d",
              kept["id"], len(body_html), len(body_md))

    if not body_md.strip():
        raise ValueError("transform produced empty markdown (body was HTML with no extractable text)")

    # Prepend metadata header. The "Article URL:" line is required so the
    # assistant's system prompt ("Cite up to 3 'Article URL:' lines per reply")
    # has something concrete to retrieve and quote back.
    header = (
        f"# {title}\n\n"
        f"Article URL: {kept['html_url']}\n"
        f"Last Updated: {kept['updated_at']}\n\n"
        "---\n\n"
    )

    filepath.write_text(header + body_md, encoding="utf-8")

    final_record = {
        "id": kept["id"],
        "slug": slug,
        "title": title,
        "url": kept["html_url"],
        "updated_at": kept["updated_at"],
        "file": filepath.name,
    }
    log.info("SAVE id=%s file=%s final_fields=%s", kept["id"], filepath.name, list(final_record.keys()))
    return final_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./articles", help="output directory for .md files")
    parser.add_argument("--limit", type=int, default=None, help="max number of articles to pull")
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="filename (inside --out) for the id/updated_at manifest, used later for delta detection",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    log.info("PULL: fetching article list from Zendesk API...")
    articles = fetch_all_articles(session, limit=args.limit)
    log.info("PULL: total articles fetched: %d", len(articles))

    manifest = []
    seen_slugs: set = set()
    skipped = 0
    failed = 0

    for article in articles:
        # One malformed/invalid article should not take down the whole run --
        # this matters most once this script is the daily unattended job.
        try:
            meta = save_article(article, out_dir, seen_slugs)
            manifest.append(meta)
        except ValueError as exc:
            # Expected, "clean" skip: validation or transform rejected it.
            skipped += 1
            log.warning("SKIP id=%s title=%r: %s", article.get("id"), article.get("title"), exc)
        except Exception as exc:
            # Unexpected: disk error, encoding error, KeyError, etc.
            failed += 1
            log.error("FAIL id=%s title=%r: %s", article.get("id"), article.get("title"), exc)

    (out_dir / args.manifest).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("DONE. saved=%d skipped=%d failed=%d fetched=%d",
              len(manifest), skipped, failed, len(articles))
    log.info("Manifest written to %s", out_dir / args.manifest)


if __name__ == "__main__":
    main()