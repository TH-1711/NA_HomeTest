FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrape.py build_assistant.py main.py ./

# /app/articles is regenerated fresh every run (re-scrape).
# /app/state must be mounted from a persistent volume so delta detection
# works across runs -- see README for how the scheduled platform mounts it.
RUN mkdir -p /app/articles /app/state
VOLUME ["/app/state"]

ENTRYPOINT ["python", "main.py"]
CMD ["--articles-dir", "/app/articles", "--state-file", "/app/state/state.json", "--build-result", "/app/state/build_result.json"]
