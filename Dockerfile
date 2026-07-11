FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrape.py build_assistant.py main.py run_forever.sh ./
RUN chmod +x run_forever.sh

# /app/articles is regenerated fresh every run (re-scrape).
# /app/state must be mounted from a persistent volume so delta detection
# works across runs -- see README for how the scheduled platform mounts it.
#
# NOTE: /app/state can only persist if the service type actually supports
# attaching a disk here. Render Cron Jobs CANNOT attach persistent disks
# (confirmed in Render's docs) -- only Web Services, Private Services, and
# Background Workers can. For Render specifically, deploy this as a
# Background Worker with a disk mounted at /app/state, and override the
# service's start command to run_forever.sh (see README_step3.md) instead
# of relying on Render's own cron scheduler. DigitalOcean App Platform's
# Scheduled Job component does not have this restriction and can use the
# default ENTRYPOINT/CMD below directly with a mounted Volume.
RUN mkdir -p /app/articles /app/state
VOLUME ["/app/state"]

ENTRYPOINT ["python", "main.py"]
CMD ["--articles-dir", "/app/articles", "--state-file", "/app/state/state.json", "--build-result", "/app/state/build_result.json"]
