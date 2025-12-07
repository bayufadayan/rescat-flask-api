import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
    TOPK = int(os.getenv("TOPK", "3"))
    PORT = int(os.getenv("PORT", "5000"))
    MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "8"))
    CORS_ENABLED = os.getenv("CORS_ENABLED", "false").lower() == "true"
    
    ONNX_PATH = os.getenv("ONNX_PATH", "models/validation_model/mobilenetv3_small.onnx")
    CLASSES_PATH = os.getenv("CLASSES_PATH", "models/validation_model/imagenet_classes.txt")
    
    CAT_HEAD_ONNX = os.getenv("CAT_HEAD_ONNX", "models/validation_model/cat_head_model.onnx")
    CAT_HEAD_CLASSES = os.getenv("CAT_HEAD_CLASSES", "models/validation_model/cat_head_classes.json")
    IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
    CONF_RAW = float(os.getenv("CONF_RAW", "0.20"))
    MIN_CONF = float(os.getenv("MIN_CONF", "0.40"))
    MID_CONF = float(os.getenv("MID_CONF", "0.50"))
    HI_COUNT = float(os.getenv("HI_COUNT", "0.75"))
    HI_PRIORITY = float(os.getenv("HI_PRIORITY", "0.80"))
    MAX_DET = int(os.getenv("MAX_DET", "5"))
    
    LANDMARK_BBOX_ONNX = os.getenv("LANDMARK_BBOX_ONNX", "models/landmark_model/frederic_bbox.onnx")
    LANDMARK_ONNX = os.getenv("LANDMARK_ONNX", "models/landmark_model/frederic_landmarks.onnx")
    
    EYE_MODEL_PATH = os.getenv("EYE_MODEL_PATH", "models/classification_model/efficientnet_cat_eye_model_b1.h5")
    EAR_MODEL_PATH = os.getenv("EAR_MODEL_PATH", "models/classification_model/efficientnet_cat_ear_model_b1.h5")
    MOUTH_MODEL_PATH = os.getenv("MOUTH_MODEL_PATH", "models/classification_model/efficientnet_cat_mouth_model_b1.h5")
    
    BUCKET_PREVIEW = os.getenv("BUCKET_PREVIEW", "preview-bounding-box")
    BUCKET_ROI = os.getenv("BUCKET_ROI", "roi-face-cat")
    BUCKET_LANDMARK = os.getenv("BUCKET_LANDMARK", "landmark-crops")
    BUCKET_LANDMARKED_FACE = os.getenv("BUCKET_LANDMARKED_FACE", "landmarked_face")
    
    CONTENT_API_BASE = os.getenv("CONTENT_API_BASE", "https://storage.rescat.life")
    CONTENT_API_TIMEOUT_CONNECT = int(os.getenv("CONTENT_API_TIMEOUT_CONNECT", "5"))
    CONTENT_API_TIMEOUT_READ = int(os.getenv("CONTENT_API_TIMEOUT_READ", "30"))
    
    REMOVEBG_CACHE_DIR = os.getenv("REMOVEBG_CACHE_DIR", "cache/remove-bg")

config = Config()
os.makedirs(config.REMOVEBG_CACHE_DIR, exist_ok=True)
