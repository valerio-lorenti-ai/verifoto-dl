FROM python:3.11-slim

ARG MODEL_URL

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y curl

RUN mkdir -p weights && \
    curl -L "$MODEL_URL" -o weights/best.pt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]