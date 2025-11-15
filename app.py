# app.py
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
from flask import Flask, request, g, redirect, url_for, render_template, send_file
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from rembg import remove
import requests
from utils.validation import validate_upload, MAX_FILE_MB
from utils.responses import ok, err
from inference import CatClassifier
from yolo_face import CatFaceDetector
from utils.uploader import upload_image_bytes  # <--- NEW

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

# Upload buckets
BUCKET_PREVIEW = os.getenv("BUCKET_PREVIEW", "preview-bounding-box")
BUCKET_ROI = os.getenv("BUCKET_ROI", "roi-face-cat")

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
    g.request_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()

@app.after_request
def after_request(resp):
    dur_ms = int((time.time() - getattr(g, "t0", time.time())) * 1000)
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    resp.headers["X-Response-Time-ms"] = str(dur_ms)
    log.info("%s %s %s %sms rid=%s",
             request.method, request.path, resp.status_code, dur_ms, getattr(g, "request_id", "-"))
    return resp

# ================== Error Handlers ==================
@app.errorhandler(413)
def request_too_large(e):
    return err("FILE_TOO_LARGE", f"Max {MAX_FILE_MB} MB", status=413)

@app.errorhandler(Exception)
def unhandled_exception(e):
    log.exception("UNHANDLED: %s", e)
    return err("INTERNAL_ERROR", "Unexpected error", status=500)

# ================== Helper Functions ==================

def read_image_bytes_from_form():
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


def _upload_artifacts(base_name: str, fr) -> dict:
    out = {"preview": None, "roi": None, "preview_error": None, "roi_error": None}

    # Preview (bounding box)
    if getattr(fr, "preview_jpeg", None):
        up = upload_image_bytes(fr.preview_jpeg, BUCKET_PREVIEW, f"{base_name}-preview.jpg")
        if up.get("ok"):
            d = up.get("data", up)
            out["preview"] = {"id": d.get("id"), "filename": d.get("filename"), "url": d.get("url")}
        else:
            out["preview_error"] = up.get("message", "Upload preview failed")

    # ROI (crop)
    if getattr(fr, "roi_jpeg", None):
        up = upload_image_bytes(fr.roi_jpeg, BUCKET_ROI, f"{base_name}-roi.jpg")
        if up.get("ok"):
            d = up.get("data", up)
            out["roi"] = {"id": d.get("id"), "filename": d.get("filename"), "url": d.get("url")}
        else:
            out["roi_error"] = up.get("message", "Upload ROI failed")

    return out



# ================== Routes ==================
@app.get("/")
def root_redirect():
    return redirect(url_for("v1_root"), code=302)

@app.get("/v1")
def v1_root():
    data = {
        "service": "ResCat API",
        "version": "v1",
        "recognizer_loaded": MODEL_READY,
        "face_loaded": FACE_READY
    }
    return render_template("index.html", data=data)

@app.get("/healthz")
def healthz():
    return ok({"recognizer_loaded": MODEL_READY, "face_loaded": FACE_READY})

@app.post("/v1/cat/recognize")
def recognize_cat():
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

    # Face detection + upload artifacts
    if FACE_READY:
        try:
            fr = app.face_detector.detect(image_bytes, include_roi_b64=False)
            base_name = f"{int(time.time()*1000)}-{g.request_id}"
            uploads = _upload_artifacts(base_name, fr)

            faces_payload = {
                "ok": fr.ok,
                "faces_count": fr.faces_count,
                "chosen_conf": fr.chosen_conf,
                "box": fr.box,
                "note": fr.note,
                "kept_confs_ge_min": fr.kept_confs_ge_min,
                "meta": fr.meta,
                # tambahkan URL hasil upload (tanpa base64)
                **uploads
            }
        except Exception as e:
            faces_payload = {"ok": False, "error": f"Face detector error: {e}"}
        payload["faces"] = faces_payload
    else:
        payload["faces"] = None

    return ok(payload)

@app.post("/v1/cat/faces")
def detect_faces():
    if not FACE_READY:
        return err("MODEL_NOT_READY", "Face detector not loaded", status=503)

    image_bytes, error_resp = read_image_bytes_from_form()
    if error_resp:
        return error_resp

    try:
        fr = app.face_detector.detect(image_bytes, include_roi_b64=False)
        base_name = f"{int(time.time()*1000)}-{g.request_id}"
        uploads = _upload_artifacts(base_name, fr)

        return ok({
            "request_id": g.request_id,
            "faces_count": fr.faces_count,
            "chosen_conf": fr.chosen_conf,
            "box": fr.box,
            "note": fr.note,
            "kept_confs_ge_min": fr.kept_confs_ge_min,
            "meta": fr.meta,
            **uploads
        })
    except Exception as e:
        return err("INFERENCE_ERROR", f"Face detector error: {e}", status=500)

@app.post("/v1/cat/remove-bg")
def remove_bg_route():
    """
    Remove background dari gambar (CPU only, unlimited, pakai rembg).

    Cara pakai:
    - MODE 1: Upload file
        POST multipart/form-data
        field: file (image)

    - MODE 2: Pakai URL
        a) JSON body: { "url": "https://..." }
        b) atau form-data: key=url, value=<image-url>
        c) atau query: /v1/cat/remove-bg?url=https://...
    """
    image_bytes = None

    # ---------- MODE 1: file upload ----------
    if "file" in request.files and request.files["file"].filename:
        image_bytes, error_resp = read_image_bytes_from_form()
        if error_resp:
            return error_resp
    else:
        # ---------- MODE 2: URL ----------
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict() or {}

        url = data.get("url") or request.args.get("url")

        if not url:
            return err(
                "INVALID_INPUT",
                "Provide either form-data 'file' or 'url' (JSON/form/query)",
                status=400,
            )

        try:
            # batas ukuran file remote (sedikit di atas MAX_FILE_MB)
            max_bytes = int((MAX_FILE_MB + 1) * 1024 * 1024)

            resp = requests.get(
                url,
                timeout=10,
                stream=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                }
            )
            if not resp.ok:
                return err(
                    "FETCH_FAILED",
                    f"Failed to fetch image from URL (status {resp.status_code})",
                    status=400,
                )

            # cek content length kalau ada
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return err(
                    "FILE_TOO_LARGE",
                    f"Remote file bigger than {MAX_FILE_MB} MB",
                    status=413,
                )

            content = resp.content
            if len(content) > max_bytes:
                return err(
                    "FILE_TOO_LARGE",
                    f"Remote file bigger than {MAX_FILE_MB} MB",
                    status=413,
                )

            # verifikasi memang image
            try:
                img = Image.open(io.BytesIO(content))
                img.verify()
            except UnidentifiedImageError:
                return err(
                    "UNSUPPORTED_MEDIA_TYPE",
                    "URL does not point to a valid image",
                    status=415,
                )

            image_bytes = content

        except Exception as e:
            log.exception("FETCH_URL_ERROR: %s", e)
            return err(
                "FETCH_URL_ERROR",
                f"Failed to download image from URL: {e}",
                status=400,
            )

    # ---------- PROSES REMOVE BACKGROUND ----------
    try:
        out_bytes = remove(image_bytes)

        buf = io.BytesIO(out_bytes)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=False,
            download_name="removed-bg.png",
        )
    except Exception as e:
        log.exception("REMOVE_BG_ERROR: %s", e)
        return err(
            "REMOVE_BG_ERROR",
            f"Failed to remove background: {e}",
            status=500,
        )

# ================== Run App ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)