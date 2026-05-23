FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir timm==1.0.26 --no-deps
RUN pip install --no-cache-dir -r requirements.txt

# Create the weights directory so the app can write the model file at startup
RUN mkdir -p weights

COPY app ./app
COPY download_model.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
