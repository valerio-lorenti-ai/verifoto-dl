FROM python:3.11-slim

ARG MODEL_URL

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision==0.18.1
RUN pip install --no-cache-dir timm==1.0.26 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p weights && \
    gdown "$MODEL_URL" -O weights/best.pt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]