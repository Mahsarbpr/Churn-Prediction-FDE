FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

COPY artifacts ./artifacts
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "churn_prediction.service:app", "--host", "0.0.0.0", "--port", "8000"]