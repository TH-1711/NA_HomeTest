# OptiBot Mini-Clone

A support bot for OptiSigns.com, built by scraping the public Help Center,
loading the content into an OpenAI vector store, and keeping it in sync
with a daily delta-upload job.

## What's in this repo

| File | Purpose |
|---|---|
| `scrape.py` | Pulls Help Center articles from the Zendesk API, converts HTML → Markdown, saves `<slug>.md` files + a manifest. |
| `build_assistant.py` | One-time setup: creates the vector store, uploads all articles, creates the `OptiBot` assistant. |
| `main.py` | Daily job: re-scrapes, hashes each article, and uploads **only the delta** (added/updated/removed) to the existing vector store. |
| `run_forever.sh` | Loop wrapper so `main.py` re-runs every 24h inside a long-lived container. |
| `Dockerfile` | Container image for both one-off local runs and the deployed daily job. |
| `requirements.txt` / `.env.sample` | Dependencies and required environment variables. |

## Setup

```bash
git clone https://github.com/TH-1711/NA_HomeTest.git
cd NA_HomeTest
pip install -r requirements.txt
cp .env.sample .env   # fill in OPENAI_API_KEY
```

## Running locally

**Step 1 — Scrape articles to Markdown**
```bash
python scrape.py --out ./articles --limit 40
```
Pulls articles from `support.optisigns.com`, validates required fields
(title, body, `html_url`, `updated_at`), converts HTML to clean Markdown,
and prepends an `Article URL:` header to each file — this is what lets the
assistant cite sources later.

**Step 2 — Build the assistant + vector store (run once)**
```bash
python build_assistant.py --articles-dir ./articles --out ./build_result.json
```
Chunk size is computed from the actual token-length distribution of the
scraped articles (not OpenAI's 800-token default) so most articles fit in
a single chunk, keeping each chunk's `Article URL:` citation line intact.
Creates the `OptiBot` assistant and vector store, saves IDs to
`build_result.json` for reuse by `main.py`.

**Step 3 — Daily delta-sync job**
```bash
python main.py --articles-dir ./articles --state-file ./state/state.json \
                --build-result ./build_result.json --limit 40
```
Re-scrapes, hashes each article (SHA-256), and uploads only what
changed — added/updated articles are uploaded, removed/updated ones have
their stale vector-store file deleted. State is tracked in `state.json`
between runs.

## Deployment (daily job)

Deployed as a Docker-based **Background Worker** on Render with a
persistent Disk at `/app/state` (Render Cron Jobs can't attach a disk, so
`run_forever.sh` loops `main.py` every 24h instead).

Daily job logs: **[insert your Render service Logs URL here]**

Verified: `BOOTSTRAP: matched 40/40 articles` → `DELTA: added=0 updated=0
skipped=40 removed=0`. Change-detection was also confirmed by corrupting a
stored hash and checking the next run correctly flagged and re-uploaded
that article.

## Sample output

**User:** How do I add a YouTube video?

**OptiBot:**
> To add a YouTube video using OptiSigns, follow these steps:
> 1. **Open the OptiSigns Portal**: Log in to your OptiSigns account.
> 2. **Navigate to Add Assets**: Go to Assets → Add Asset and select the YouTube option.
> 3. **Input Video Details**: Enter the YouTube URL or embed code for the video you wish to add.
> 4. **Set Display Options**: Configure how you want the video to be displayed on your screens, including dimensions and display duration.
> 5. **Save and Deploy**: Save your changes and assign the asset to your screens to start displaying the video.
>
> For detailed instructions, refer to the article here: [How to Use the YouTube Dashboard App](https://support.optisigns.com/hc/en-us/articles/48626115821459-How-to-Use-the-YouTube-Dashboard-App) `[1]`

**User:** How to Install a Root Certificate and Display an Internal Website on Screens

**OptiBot:**
> To install a root certificate and display an internal website on your screens, follow these steps:
>
> **Installing a Root Certificate:**
> 1. **Choose Your Device**: Different instructions are available for OptiSigns Gen 3 Pro, Linux/Ubuntu, Windows, and MacOS.
> 2. **Ensure Valid Format**: For Gen 3 Pro Player, your certificate must have a `.crt` extension. Rename your `.pem` file to `.crt`.
> 3. **Import the Certificate**: Use the relevant method for your operating system to import the certificate.
> 4. **Verify Installation**: Check if the certificate is successfully installed by accessing the internal website.
>
> **Displaying the Internal Website:**
> 1. **Use the Website App**: Go to Files/Assets in your OptiSigns portal and select the Website app.
> 2. **Input URL**: Enter the URL of your internal webpage.
> 3. **Assign the Asset**: Once created, assign the Website asset to your screens.
>
> For complete instructions, see the full article here: [How to Install a Root Certificate and Display an Internal Website on Screens](https://support.optisigns.com/hc/en-us/articles/35184720136595-How-to-Install-a-Root-Certificate-and-Display-an-Internal-Website-on-Screens) `[1]`

Screenshot: `screenshots/sanity_check.png`