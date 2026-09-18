FROM python:3.11-slim

WORKDIR /app

COPY requirements-vessel.txt .
RUN pip install --no-cache-dir -r requirements-vessel.txt

COPY shared/ shared/
COPY vessel/ vessel/

ENV PYTHONPATH=/app

CMD ["uvicorn", "vessel.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
