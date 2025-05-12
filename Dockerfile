FROM python:3.10-slim

WORKDIR /app

# Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY ./app /app/app

# Şu anda model dosyası yok, daha sonra eklenecek
RUN mkdir -p /app/app/models

# Uygulama başlatma
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"] 