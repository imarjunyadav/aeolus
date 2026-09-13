FROM python:3.11-slim

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY shared/ shared/
COPY cloud/ cloud/
COPY data/ data/

ENV PYTHONPATH=/app

CMD ["uvicorn", "cloud.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
