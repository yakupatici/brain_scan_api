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

# Model yolu
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "InceptionOnnx.onnx")

# Global model nesnesi
ort_session = None

@app.on_event("startup")
async def startup_event():
    """API başlatıldığında modeli yükle"""
    global ort_session
    try:
        logger.info(f"ONNX Runtime sürümü: {ort.__version__}")
        logger.info(f"Model yükleniyor: {MODEL_PATH}")
        
        # Model dosyasının varlığını kontrol et
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
            logger.info("Çalışma dizini: " + os.getcwd())
            logger.info("Dizin içeriği: " + str(os.listdir(os.path.dirname(MODEL_PATH))))
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        
        # Modeli yükle
        start_time = time.time()
        ort_session = ort.InferenceSession(MODEL_PATH)
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
    """Beyin taraması görüntüsünü analiz et"""
    # Model kontrolü
    if ort_session is None:
        logger.error("Model yüklü değil, tahmin yapılamıyor")
        raise HTTPException(status_code=503, detail="Model yüklü değil, lütfen daha sonra tekrar deneyin")
    
    start_time = time.time()
    
    try:
        logger.info(f"Dosya alındı: {file.filename}, {file.content_type}")
        
        # Dosya içeriğini oku
        contents = await file.read()
        
        if len(contents) == 0:
            logger.error("Boş dosya yüklendi")
            raise HTTPException(status_code=400, detail="Boş dosya yüklendi")
        
        # Görüntüyü ön işle
        try:
            logger.info("Görüntü ön işleniyor")
            img_processed = preprocess_image(contents)
            logger.info(f"Görüntü ön işlendi, şekil: {img_processed.shape}")
        except ValueError as e:
            logger.error(f"Görüntü işleme hatası: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # ONNX modeliyle tahminde bulun
        try:
            logger.info("Model tahmini yapılıyor")
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: img_processed})
            logger.info(f"Tahmin tamamlandı, çıktı şekli: {outputs[0].shape}")
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model tahmini başarısız: {str(e)}")
        
        # Sonuçları işle
        scores = outputs[0][0]
        logger.info(f"Ham skorlar: {scores}")
        
        # Softmax uygula (isteğe bağlı)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # En yüksek olasılıklı sınıfı bul
        predicted_class_idx = np.argmax(probs)
        prediction = CLASS_NAMES[predicted_class_idx]
        confidence = float(probs[predicted_class_idx] * 100)
        
        # Tüm sınıflar için skorlar
        all_scores = {CLASS_NAMES[i]: float(probs[i] * 100) for i in range(len(CLASS_NAMES))}
        
        processing_time = time.time() - start_time
        logger.info(f"Tahmin: {prediction}, Güven: {confidence:.1f}%, İşlem süresi: {processing_time:.2f} saniye")
        
        # Sonucu döndür
        return JSONResponse({
            "prediction": prediction,
            "confidence": confidence,
            "scores": all_scores,
            "processing_time_ms": round(processing_time * 1000)
        })
    
    except Exception as e:
        logger.error(f"İşlem hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}") 