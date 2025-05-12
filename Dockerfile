FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

RUN mkdir -p /app/app/models

RUN echo '#!/bin/bash' > /app/start.sh
RUN echo 'PORT=${PORT:-8000}' >> /app/start.sh
RUN echo 'exec uvicorn app.main:app --host 0.0.0.0 --port $PORT' >> /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
