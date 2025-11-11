"""
ResCat API — Flask-based service for cat recognition and face detection.

Features:
- Recognize whether an image contains a cat
- Detect cat faces and optionally return ROI
- Return structured JSON responses with metadata
- Health endpoint at /healthz
"""

import os
import uuid
import time
import logging
import io
from flask import Flask, request, g, redirect, url_for, render_template
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from utils.validation import validate_upload, MAX_FILE_MB
from utils.responses import ok, err
from inference import CatClassifier
from yolo_face import CatFaceDetector

# Optional CORS
CORS_ENABLED = os.getenv("CORS_ENABLED", "false").lower() == "true"
if CORS_ENABLED:
    from flask_cors import CORS

# Load environment variables
load_dotenv()

# ================== Configuration ==================
# Recognition
THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
TOPK = int(os.getenv("TOPK", "3"))
PORT = int(os.getenv("PORT", "8000"))

# Face detection (YOLO)
CAT_HEAD_ONNX = os.getenv("CAT_HEAD_ONNX", "models/cat_head_model.onnx")
CAT_HEAD_CLASSES = os.getenv("CAT_HEAD_CLASSES", "models/cat_head_classes.json")
IMG_SIZE = int(os.getenv("IMG_SIZE", "768"))
CONF_RAW = float(os.getenv("CONF_RAW", "0.20"))
MIN_CONF = float(os.getenv("MIN_CONF", "0.40"))
MID_CONF = float(os.getenv("MID_CONF", "0.50"))
HI_COUNT = float(os.getenv("HI_COUNT", "0.75"))
HI_PRIORITY = float(os.getenv("HI_PRIORITY", "0.80"))
MAX_DET = int(os.getenv("MAX_DET", "5"))

# ================== Logger ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("cat-api")

# ================== Flask App ==================
app = Flask(__name__)
if CORS_ENABLED:
    CORS(app)

# Limit request size slightly above MAX_FILE_MB
app.config["MAX_CONTENT_LENGTH"] = int((MAX_FILE_MB + 1) * 1024 * 1024)

# ================== Model Loading ==================
MODEL_READY = False
FACE_READY = False

try:
    app.classifier = CatClassifier(
        onnx_path=os.getenv("ONNX_PATH", "models/mobilenetv3_small.onnx"),
        classes_path=os.getenv("CLASSES_PATH", "models/imagenet_classes.txt"),
        threshold=THRESHOLD,
        topk=TOPK,
    )
    MODEL_READY = True
    log.info("Classifier loaded successfully")
except Exception as e:
    log.exception("MODEL INIT ERROR: %s", e)

try:
    app.face_detector = CatFaceDetector(
        onnx_path=CAT_HEAD_ONNX,
        classes_path=CAT_HEAD_CLASSES,
        img_size=IMG_SIZE,
        conf_raw=CONF_RAW,
        min_conf=MIN_CONF,
        mid_conf=MID_CONF,
        hi_count=HI_COUNT,
        hi_priority=HI_PRIORITY,
        max_det=MAX_DET
    )
    FACE_READY = True
    log.info("Face detector loaded successfully")
except Exception as e:
    log.exception("FACE MODEL INIT ERROR: %s", e)

# ================== Request Lifecycle ==================
@app.before_request
def before_request():
    """Generate request ID and start timer."""
    g.request_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()

@app.after_request
def after_request(resp):
    """Attach request metadata (X-Request-ID, latency)."""
    dur_ms = int((time.time() - getattr(g, "t0", time.time())) * 1000)
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    resp.headers["X-Response-Time-ms"] = str(dur_ms)
    log.info("%s %s %s %sms rid=%s",
             request.method, request.path, resp.status_code, dur_ms, getattr(g, "request_id", "-"))
    return resp

# ================== Error Handlers ==================
@app.errorhandler(413)
def request_too_large(e):
    """Return error if file exceeds limit."""
    return err("FILE_TOO_LARGE", f"Max {MAX_FILE_MB} MB", status=413)

@app.errorhandler(Exception)
def unhandled_exception(e):
    """Global exception handler."""
    log.exception("UNHANDLED: %s", e)
    return err("INTERNAL_ERROR", "Unexpected error", status=500)

# ================== Helper Functions ==================
def read_image_bytes_from_form():
    """
    Read and validate uploaded image from form-data field 'file'.
    Returns: (bytes, error_response)
    """
    if "file" not in request.files:
        return None, err("INVALID_FILE", "Expect form-data field 'file'")
    
    file = request.files["file"]
    is_valid, code, msg = validate_upload(file)
    if not is_valid:
        status = 415 if code == "UNSUPPORTED_MEDIA_TYPE" else 400
        return None, err(code, msg, status=status)

    try:
        image_bytes = file.read()
        _img = Image.open(io.BytesIO(image_bytes))
        _img.verify()
        return image_bytes, None
    except UnidentifiedImageError:
        return None, err("UNSUPPORTED_MEDIA_TYPE", "Not a valid image", status=415)
    except Exception as e:
        return None, err("INVALID_FILE", f"Failed to read image: {e}", status=400)

def human_message(is_cat: bool, face_ok: bool) -> str:
    """Return human-friendly detection message."""
    if not is_cat and not face_ok:
        return "Kami tidak mendeteksi kucing."
    if is_cat and not face_ok:
        return "Wajah kurang terlihat jelas. Pastikan pencahayaan baik dan area cukup luas."
    return "Deteksi valid."

# ================== Routes ==================
@app.get("/")
def root_redirect():
    """Redirect to /v1 root."""
    return redirect(url_for("v1_root"), code=302)

@app.get("/v1")
def v1_root():
    """Return service status info for API root."""
    data = {
        "service": "ResCat API",
        "version": "v1",
        "recognizer_loaded": MODEL_READY,
        "face_loaded": FACE_READY
    }
    return render_template("index.html", data=data)

@app.get("/healthz")
def healthz():
    """Simple health check endpoint."""
    return ok({"recognizer_loaded": MODEL_READY, "face_loaded": FACE_READY})

@app.post("/v1/cat/recognize")
def recognize_cat():
    """POST image → classify cat + optional face detection."""
    if not MODEL_READY:
        return err("MODEL_NOT_READY", "Classifier not loaded", status=503)
    if not FACE_READY:
        log.warning("Face detector not ready; skipping face analysis.")

    image_bytes, error_resp = read_image_bytes_from_form()
    if error_resp:
        return error_resp

    try:
        res = app.classifier.predict(image_bytes)
    except Exception as e:
        return err("INFERENCE_ERROR", f"Classifier error: {e}", status=500)

    payload = {
        "ok": True,
        "request_id": g.request_id,
        "label": res.label,
        "cat_prob": res.cat_prob,
        "threshold": res.threshold,
        "topk": res.topk,
        "meta": {**res.meta, "api_latency_ms": int((time.time() - g.t0) * 1000)}
    }

    # Optional face detection chaining
    faces_payload = None
    if FACE_READY:
        try:
            fr = app.face_detector.detect(image_bytes, include_roi_b64=True)
            faces_payload = {
                "ok": fr.ok,
                "faces_count": fr.faces_count,
                "chosen_conf": fr.chosen_conf,
                "box": fr.box,
                "note": fr.note,
                "kept_confs_ge_min": fr.kept_confs_ge_min,
                "meta": fr.meta,
                "roi_b64": fr.roi_b64
            }
        except Exception as e:
            faces_payload = {"ok": False, "error": f"Face detector error: {e}"}
    payload["faces"] = faces_payload

    return ok(payload)

@app.post("/v1/cat/faces")
def detect_faces():
    """POST image → detect faces only."""
    if not FACE_READY:
        return err("MODEL_NOT_READY", "Face detector not loaded", status=503)

    image_bytes, error_resp = read_image_bytes_from_form()
    if error_resp:
        return error_resp

    try:
        fr = app.face_detector.detect(image_bytes, include_roi_b64=True)
        return ok({
            "request_id": g.request_id,
            "faces_count": fr.faces_count,
            "chosen_conf": fr.chosen_conf,
            "box": fr.box,
            "note": fr.note,
            "kept_confs_ge_min": fr.kept_confs_ge_min,
            "meta": fr.meta,
            "roi_b64": fr.roi_b64
        })
    except Exception as e:
        return err("INFERENCE_ERROR", f"Face detector error: {e}", status=500)

# ================== Run App ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
