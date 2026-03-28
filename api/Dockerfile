FROM python:3.11-slim
ARG MODEL_URL

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p weights && \
    apt-get update && apt-get install -y curl && \
    curl -L "$MODEL_URL" -o weights/best.pt
    
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]