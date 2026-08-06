FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /scraper

COPY scraper/requirements.txt /scraper/requirements.txt
RUN pip install --no-cache-dir -r /scraper/requirements.txt

COPY scraper /scraper

# APScheduler runs jobs; entrypoint overridden per docker-compose command
CMD ["sleep", "infinity"]
