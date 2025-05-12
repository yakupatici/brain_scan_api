# Brain Scan API (Backend)

Bu proje, beyin taramalarını analiz etmek için ONNX formatında bir derin öğrenme modeli kullanan FastAPI tabanlı bir API sunar.

## Özellikler

- Beyin taramaları üzerinde tümör, enfarkt ve kanama analizi
- ONNX Runtime ile optimize edilmiş model çalıştırma
- FastAPI ile modern ve hızlı REST API
- CORS desteği ile farklı kaynaklardan erişim
- Detaylı hata işleme ve loglama
- Docker ile kolay deployment

## Kurulum ve Başlatma

### Yerel Geliştirme

1. Depoyu klonlayın:

```bash
git clone <repo-url>
cd brain_scan_api
```

2. Sanal ortam oluşturun ve bağımlılıkları yükleyin:

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt
```

3. ONNX modelini ekleyin:
   - `InceptionOnnx.onnx` modelini `app/models/` klasörüne kopyalayın

4. API'yi başlatın:

```bash
uvicorn app.main:app --reload
```

API http://localhost:8000 adresinde çalışacaktır.

### Docker ile Çalıştırma

```bash
docker build -t brain-scan-api .
docker run -p 8000:8000 brain-scan-api
```

## API Kullanımı

### Sağlık Kontrolü
```
GET /
```

Yanıt:
```json
{
  "message": "Brain Scan Analysis API aktif",
  "model_loaded": true
}
```

### Tahmin
```
POST /predict
```

İstek: `multipart/form-data` ile bir görüntü dosyası yükleyin.

Yanıt:
```json
{
  "prediction": "Sağlıklı",
  "confidence": 95.7,
  "scores": {
    "Sağlıklı": 95.7,
    "Enfarkt": 2.1,
    "Tümör": 1.5,
    "Kanama": 0.7
  },
  "processing_time_ms": 156
}
```

## Deployment

Bu API Railway.app'te barındırılmak üzere tasarlanmıştır. `railway.json` ve `Dockerfile` dosyaları deployment için gerekli yapılandırmaları içerir.

### Railway.app Deployment Adımları

1. [Railway CLI](https://docs.railway.app/develop/cli) yükleyin
2. Railway hesabınızda oturum açın: `railway login`
3. Projeyi başlatın: `railway init`
4. Railway.app'e deploy edin: `railway up`

## InceptionOnnx Modeli Hakkında

Bu API, beyin taramalarını analiz etmek için Inception mimarisi kullanılarak eğitilmiş ve ONNX formatına dönüştürülmüş bir derin öğrenme modeli kullanmaktadır. Model, beyin BT görüntülerini dört sınıftan birine atayabilir:

1. Sağlıklı
2. Enfarkt
3. Tümör
4. Kanama

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 