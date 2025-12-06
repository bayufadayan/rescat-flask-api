"""ResCat API - Flask service for cat recognition and face detection."""

import os
import uuid
import time
import logging
import io
import hashlib
import json
from flask import Flask, request, g, redirect, url_for, render_template
from PIL import Image, UnidentifiedImageError
from rembg import remove
import requests

from config import config
from utils.validation import validate_upload
from utils.responses import ok, err
from inference import CatClassifier
from yolo_face import CatFaceDetector
from landmark_detector import LandmarkDetector
from utils.uploader import upload_image_bytes

if config.CORS_ENABLED:
    from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cat-api")

app = Flask(__name__)
if config.CORS_ENABLED:
    CORS(app)

app.config["MAX_CONTENT_LENGTH"] = int((config.MAX_FILE_MB + 1) * 1024 * 1024)

MODEL_READY = False
FACE_READY = False
LANDMARK_READY = False

try:
    app.classifier = CatClassifier(
        onnx_path=config.ONNX_PATH,
        classes_path=config.CLASSES_PATH,
        threshold=config.THRESHOLD,
        topk=config.TOPK,
    )
    MODEL_READY = True
    log.info("Classifier loaded")
except Exception as e:
    log.exception("MODEL INIT ERROR: %s", e)

try:
    app.face_detector = CatFaceDetector(
        onnx_path=config.CAT_HEAD_ONNX,
        classes_path=config.CAT_HEAD_CLASSES,
        img_size=config.IMG_SIZE,
        conf_raw=config.CONF_RAW,
        min_conf=config.MIN_CONF,
        mid_conf=config.MID_CONF,
        hi_count=config.HI_COUNT,
        hi_priority=config.HI_PRIORITY,
        max_det=config.MAX_DET
    )
    FACE_READY = True
    log.info("Face detector loaded")
except Exception as e:
    log.exception("FACE MODEL INIT ERROR: %s", e)

try:
    app.landmark_detector = LandmarkDetector(
        bbox_onnx_path=config.LANDMARK_BBOX_ONNX,
        landmark_onnx_path=config.LANDMARK_ONNX
    )
    LANDMARK_READY = True
    log.info("Landmark detector loaded")
except Exception as e:
    log.exception("LANDMARK MODEL INIT ERROR: %s", e)

@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())[:8]
    g.t0 = time.time()

@app.after_request
def after_request(resp):
    dur_ms = int((time.time() - getattr(g, "t0", time.time())) * 1000)
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    resp.headers["X-Response-Time-ms"] = str(dur_ms)
    log.info("%s %s %s %sms rid=%s", request.method, request.path, resp.status_code, dur_ms, getattr(g, "request_id", "-"))
    return resp

@app.errorhandler(413)
def request_too_large(e):
    return err("FILE_TOO_LARGE", f"Max {config.MAX_FILE_MB} MB", status=413)

@app.errorhandler(Exception)
def unhandled_exception(e):
    log.exception("UNHANDLED: %s", e)
    return err("INTERNAL_ERROR", "Unexpected error", status=500)

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_image_bytes_from_form():
    log.info(f"Content-Type: {request.content_type}")
    log.info(f"request.files keys: {list(request.files.keys())}")
    log.info(f"request.form keys: {list(request.form.keys())}")
    log.info(f"request.headers: {dict(request.headers)}")
    
    if "file" not in request.files:
        available_keys = list(request.files.keys())
        log.warning(f"'file' key not found. Available keys: {available_keys}")
        return None, err("INVALID_FILE", f"Expect form-data field 'file'. Found: {available_keys}")
    
    file = request.files["file"]
    if not file.filename:
        return None, err("INVALID_FILE", "No file selected")
    
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

    if getattr(fr, "preview_jpeg", None):
        up = upload_image_bytes(fr.preview_jpeg, config.BUCKET_PREVIEW, f"{base_name}-preview.jpg")
        if up.get("ok"):
            d = up.get("data", up)
            out["preview"] = {"id": d.get("id"), "filename": d.get("filename"), "url": d.get("url")}
        else:
            out["preview_error"] = up.get("message", "Upload preview failed")

    if getattr(fr, "roi_jpeg", None):
        up = upload_image_bytes(fr.roi_jpeg, config.BUCKET_ROI, f"{base_name}-roi.jpg")
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
        "face_loaded": FACE_READY,
        "landmark_loaded": LANDMARK_READY
    }
    return render_template("index.html", data=data)

@app.get("/healthz")
def healthz():
    return ok({
        "recognizer_loaded": MODEL_READY,
        "face_loaded": FACE_READY,
        "landmark_loaded": LANDMARK_READY,
    })

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
    """Remove background from image using rembg."""
    image_bytes = None

    if "file" in request.files and request.files["file"].filename:
        image_bytes, error_resp = read_image_bytes_from_form()
        if error_resp:
            return error_resp
    else:
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
            max_bytes = int((config.MAX_FILE_MB + 1) * 1024 * 1024)

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
                },
            )
            if not resp.ok:
                return err(
                    "FETCH_FAILED",
                    f"Failed to fetch image from URL (status {resp.status_code})",
                    status=400,
                )

            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return err(
                    "FILE_TOO_LARGE",
                    f"Remote file bigger than {config.MAX_FILE_MB} MB",
                    status=413,
                )

            content = resp.content
            if len(content) > max_bytes:
                return err(
                    "FILE_TOO_LARGE",
                    f"Remote file bigger than {config.MAX_FILE_MB} MB",
                    status=413,
                )

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

    img_hash = hash_bytes(image_bytes)
    cache_path = os.path.join(config.REMOVEBG_CACHE_DIR, f"{img_hash}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_out = {
                **meta,
                "hash": img_hash,
                "cached": True,
            }
            return ok(meta_out)
        except Exception as e:
            log.warning("Failed to read cache file %s: %s", cache_path, e)

    try:
        out_bytes = remove(image_bytes)
    except Exception as e:
        log.exception("REMOVE_BG_ERROR: %s", e)
        return err(
            "REMOVE_BG_ERROR",
            f"Failed to remove background: {e}",
            status=500,
        )

    try:
        filename = f"{int(time.time() * 1000)}-{g.request_id}.png"
        upload_result = upload_image_bytes(
            out_bytes,
            "remove-bg",
            filename
        )

        if not upload_result.get("ok"):
            msg = upload_result.get("message", "Upload failed")
            return err("UPLOAD_FAILED", msg, status=502)

        data = upload_result.get("data", upload_result)

        file_id = data.get("id")
        bucket = data.get("bucket", "remove-bg")
        url = data.get("url")
        stored_filename = data.get("filename", filename)

        if not file_id or not url:
            return err(
                "UPLOAD_INVALID_RESPONSE",
                "Upload service did not return id/url",
                status=502,
            )

        meta_to_cache = {
            "id": file_id,
            "bucket": bucket,
            "url": url,
            "filename": stored_filename,
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(meta_to_cache, f)
        except Exception as e:
            log.warning("Failed to write cache file %s: %s", cache_path, e)

        meta_out = {
            **meta_to_cache,
            "hash": img_hash,
            "cached": False,
        }
        return ok(meta_out)

    except Exception as e:
        log.exception("UPLOAD_ERROR: %s", e)
        return err(
            "UPLOAD_ERROR",
            f"Failed to upload removed-bg image: {e}",
            status=502,
        )

@app.post("/v1/cat/landmark")
def landmark_route():
    """Detect cat facial landmarks and crop regions (eyes, ears, mouth)."""
    if not LANDMARK_READY:
        return err("MODEL_NOT_READY", "Landmark detector not initialized", status=503)
    
    image_bytes = None
    
    if "file" in request.files and request.files["file"].filename:
        image_bytes, error_resp = read_image_bytes_from_form()
        if error_resp:
            return error_resp
    elif request.is_json and request.json and "file" in request.json:
        file_url = request.json["file"]
        if not isinstance(file_url, str) or not file_url.strip():
            return err("INVALID_URL", "URL must be a non-empty string", status=400)
        
        try:
            resp = requests.get(file_url, timeout=15)
            resp.raise_for_status()
            content = resp.content
            
            try:
                Image.open(io.BytesIO(content)).verify()
            except UnidentifiedImageError:
                return err("UNSUPPORTED_MEDIA_TYPE", "URL does not point to a valid image", status=415)
            
            image_bytes = content
        except Exception as e:
            log.exception("FETCH_URL_ERROR: %s", e)
            return err("FETCH_URL_ERROR", f"Failed to download image from URL: {e}", status=400)
    else:
        return err("MISSING_INPUT", "Provide 'file' in form-data or JSON with image URL", status=400)
    
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return err("IMAGE_DECODE_ERROR", f"Failed to decode image: {e}", status=400)
    
    try:
        result = app.landmark_detector.detect_and_crop(img)
    except ValueError as e:
        return err("VALIDATION_ERROR", str(e), status=400)
    except Exception as e:
        log.exception("LANDMARK_DETECTION_ERROR: %s", e)
        return err("INFERENCE_ERROR", f"Landmark detection error: {e}", status=500)
    
    crops = {
        "right_eye": result.right_eye,
        "left_eye": result.left_eye,
        "mouth": result.mouth,
        "right_ear": result.right_ear,
        "left_ear": result.left_ear,
    }
    
    base_name = f"{int(time.time()*1000)}-{g.request_id}"
    uploaded_urls = {}
    
    for name, crop_bytes in crops.items():
        try:
            bucket_name = f"{name}_crop"
            filename = f"{base_name}.jpg"
            upload_result = upload_image_bytes(crop_bytes, bucket_name, filename)
            
            if not upload_result.get("ok"):
                msg = upload_result.get("message", "Upload failed")
                return err("UPLOAD_ERROR", f"Failed to upload {name}: {msg}", status=502)
            
            data = upload_result.get("data") or upload_result
            url = data.get("url")
            
            if not url:
                return err("UPLOAD_INVALID_RESPONSE", f"Upload service did not return url for {name}", status=502)
            
            uploaded_urls[bucket_name] = url
        except Exception as e:
            log.exception("UPLOAD_ERROR for %s: %s", name, e)
            return err("UPLOAD_ERROR", f"Failed to upload {name}: {e}", status=502)
    
    return ok({
        "right_eye_crop": uploaded_urls["right_eye_crop"],
        "left_eye_crop": uploaded_urls["left_eye_crop"],
        "mouth_crop": uploaded_urls["mouth_crop"],
        "right_ear_crop": uploaded_urls["right_ear_crop"],
        "left_ear_crop": uploaded_urls["left_ear_crop"],
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=True)