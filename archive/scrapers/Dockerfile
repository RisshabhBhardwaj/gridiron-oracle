FROM python:3.12-slim
WORKDIR /scraper
COPY scrapers/requirements.txt /scraper/requirements.txt
RUN pip install --no-cache-dir -r /scraper/requirements.txt
COPY scrapers /scraper
# default command does nothing; jobs run via cron/APScheduler later
CMD ["sleep", "infinity"]

