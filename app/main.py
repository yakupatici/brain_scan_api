import os
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort
import time
import requests
import urllib.request
from pathlib import Path

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("brain-scan-api")

# FastAPI uygulaması oluştur
app = FastAPI(title="Brain Scan Analysis API")

# CORS ayarları - tüm kaynaklardan isteklere izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (üretimde spesifik bir domain belirle)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sınıf etiketleri
CLASS_NAMES = ["Sağlıklı", "Enfarkt", "Tümör", "Kanama"]

# Model yolu için çevre değişkenini kullanabilirsiniz
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "InceptionOnnx.onnx"))

# Model URL'si (Google Drive veya başka bir yerden)
MODEL_URL = os.environ.get("MODEL_URL", "https://drive.google.com/uc?export=download&id=1ghlzq6mfJU6iWA_5iZWHkW0q_NXEwXKR")

# Global model nesnesi
ort_session = None

def download_model(url, save_path):
    """Verilen URL'den model dosyasını indir ve belirtilen yola kaydet"""
    try:
        logger.info(f"Model indiriliyor: {url}")
        
        # Dizini oluştur
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Google Drive için özel işleme
        if "drive.google.com" in url:
            # Google Drive URL'sinden dosyayı indir
            response = requests.get(url, stream=True)
            file_size = int(response.headers.get("Content-Length", 0))
            
            # Akış durumunu log'la
            downloaded = 0
            logger.info(f"Dosya boyutu: {file_size / (1024*1024):.2f} MB")
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        logger.info(f"İndiriliyor: {downloaded / (1024*1024):.2f} MB / {file_size / (1024*1024):.2f} MB")
        else:
            # Normal URL için direkt indir
            urllib.request.urlretrieve(url, save_path)
        
        logger.info(f"Model başarıyla indirildi: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Model indirilirken hata: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """API başlatıldığında modeli yükle"""
    global ort_session
    try:
        logger.info(f"ONNX Runtime sürümü: {ort.__version__}")
        logger.info(f"Model yolu: {MODEL_PATH}")
        
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model dosyası bulunamadı: {MODEL_PATH}")
            
            # Çalışma ortamı bilgisini logla
            logger.info("Çalışma dizini: " + os.getcwd())
            
            model_dir = os.path.dirname(MODEL_PATH)
            if os.path.exists(model_dir):
                logger.info("Model dizini mevcut. İçerik: " + str(os.listdir(model_dir)))
            else:
                logger.info(f"Model dizini mevcut değil: {model_dir}")
            
            # Model dosyasını indir
            if MODEL_URL:
                logger.info(f"Model URL belirtildi, indirme deneniyor: {MODEL_URL}")
                success = download_model(MODEL_URL, MODEL_PATH)
                if not success:
                    raise FileNotFoundError(f"Model dosyası indirilemedi: {MODEL_URL}")
            else:
                raise FileNotFoundError(f"Model dosyası bulunamadı ve MODEL_URL belirtilmedi")
        
        # Modeli yükle
        start_time = time.time()
        logger.info(f"Model yükleniyor: {MODEL_PATH}, Dosya boyutu: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        load_time = time.time() - start_time
        logger.info(f"Model başarıyla yüklendi ({load_time:.2f} saniye)")
        
        # Model giriş ve çıkış bilgilerini logla
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        output_shape = ort_session.get_outputs()[0].shape
        
        logger.info(f"Model giriş adı: {input_name}, şekli: {input_shape}")
        logger.info(f"Model çıkış adı: {output_name}, şekli: {output_shape}")
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        # Uygulama başlatılmasına izin ver, ancak endpoint'ler hatalar döndürecek
        ort_session = None

@app.get("/")
async def root():
    """Sağlık kontrolü için endpoint"""
    return {"message": "Brain Scan Analysis API aktif", "model_loaded": ort_session is not None}

@app.get("/health")
async def health():
    """API sağlık durumu için endpoint"""
    return {
        "status": "healthy" if ort_session is not None else "not_ready",
        "model_loaded": ort_session is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False
    }

def preprocess_image(image_bytes, target_size=(299, 299)):
    """Görüntüyü modele uygun formata dönüştür"""
    try:
        # Görüntüyü yükle
        image = Image.open(io.BytesIO(image_bytes))
        
        # RGB'ye dönüştür (grayscale ise)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Boyutlandır
        image = image.resize(target_size)
        
        # Numpy dizisine dönüştür
        img_array = np.array(image)
        
        # Normalleştir (0-1 aralığına)
        img_array = img_array.astype("float32") / 255.0
        
        # Inception modeli için gerekli normalleştirme
        img_array = (img_array - 0.5) * 2
        
        # Batch boyutu ekle (1, 299, 299, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # ONNX için kanal sıralamasını değiştir
        # (1, 299, 299, 3) -> (1, 3, 299, 299)
        img_array = np.transpose(img_array, (0, 3, 1, 2))
        
        return img_array
    except UnidentifiedImageError:
        raise ValueError("Geçersiz görüntü formatı")
    except Exception as e:
        raise ValueError(f"Görüntü işlenirken hata: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Beyin taraması görüntüsünü analiz et (ŞİMDİLİK BASİTLEŞTİRİLDİ)"""
    logger.info(f"--- /predict isteği alındı: {file.filename} ---") # <-- YENİ LOG

    # Model kontrolü - ŞİMDİLİK DEVRE DIŞI
    # if ort_session is None:
    #     logger.error("Model yüklü değil, tahmin yapılamıyor")
    #     # ... (rest of the original model check code) ...
    #     if ort_session is None:
    #         raise HTTPException(status_code=503, detail="Model yüklü değil, lütfen daha sonra tekrar deneyin")

    start_time = time.time()

    try:
        logger.info(f"Dosya bilgisi: {file.filename}, {file.content_type}")

        # --- GEÇİCİ OLARAK DEVRE DIŞI BIRAKILAN KOD ---
        # contents = await file.read()
        # if len(contents) == 0:
        #     logger.error("Boş dosya yüklendi")
        #     raise HTTPException(status_code=400, detail="Boş dosya yüklendi")
        #
        # try:
        #     logger.info("Görüntü ön işleniyor")
        #     img_processed = preprocess_image(contents)
        #     logger.info(f"Görüntü ön işlendi, şekil: {img_processed.shape}")
        # except ValueError as e:
        #     logger.error(f"Görüntü işleme hatası: {str(e)}")
        #     raise HTTPException(status_code=400, detail=str(e))
        #
        # try:
        #     logger.info("Model tahmini yapılıyor")
        #     input_name = ort_session.get_inputs()[0].name
        #     outputs = ort_session.run(None, {input_name: img_processed})
        #     logger.info(f"Tahmin tamamlandı, çıktı şekli: {outputs[0].shape}")
        # except Exception as e:
        #     logger.error(f"Tahmin hatası: {str(e)}")
        #     raise HTTPException(status_code=500, detail=f"Model tahmini başarısız: {str(e)}")
        #
        # scores = outputs[0][0]
        # logger.info(f"Ham skorlar: {scores}")
        # exp_scores = np.exp(scores - np.max(scores))
        # probs = exp_scores / exp_scores.sum()
        # predicted_class_idx = np.argmax(probs)
        # prediction = CLASS_NAMES[predicted_class_idx]
        # confidence = float(probs[predicted_class_idx] * 100)
        # all_scores = {CLASS_NAMES[i]: float(probs[i] * 100) for i in range(len(CLASS_NAMES))}
        # --- DEVRE DIŞI KOD SONU ---

        processing_time = time.time() - start_time
        logger.info(f"--- /predict isteği başarıyla işlendi (basitleştirilmiş mod) ---") # <-- YENİ LOG

        # Geçici, basit yanıt döndür
        return JSONResponse({
            "prediction": "GEÇİCİ YANIT",
            "confidence": 99.9,
            "scores": {k: 25.0 for k in CLASS_NAMES},
            "processing_time_ms": round(processing_time * 1000)
        })

    except Exception as e:
        # Bu blok normalde devre dışı bırakılan kod içindeki hataları yakalar
        logger.error(f"İşlem hatası (beklenmeyen): {str(e)}")
        raise HTTPException(status_code=500, detail=f"İşlem hatası (beklenmeyen): {str(e)}") 