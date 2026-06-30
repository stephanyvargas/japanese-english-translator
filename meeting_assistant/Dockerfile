FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# Cloud Run injects $PORT (defaults to 8080); honor it via shell form.
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
