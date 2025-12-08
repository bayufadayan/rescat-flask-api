"""Cat face detector using ONNXRuntime + OpenCV (CPU-only).

This module provides the same public API as the previous YOLO/ultralytics-based
implementation (`CatFaceDetector.detect` signature and `FaceResult` dataclass),
but runs entirely on CPU using an exported ONNX detection model.
"""

import base64
import io
import os
from dataclasses import dataclass
from typing import Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class FaceResult:
    ok: bool
    faces_count: int
    chosen_conf: float | None
    box: list[int] | None
    note: str
    kept_confs_ge_min: list[float]
    meta: Dict[str, Any]
    preview_jpeg: bytes | None
    roi_jpeg: bytes | None
    roi_b64: str | None


class CatFaceDetector:
    """ONNXRuntime-based cat face detector with smart ROI selection (CPU-only)."""

    def __init__(
        self,
        onnx_path: str,
        classes_path: str | None = None,
        img_size: int = 768,
        conf_raw: float = 0.20,
        min_conf: float = 0.40,
        mid_conf: float = 0.50,
        hi_count: float = 0.75,
        hi_priority: float = 0.80,
        max_det: int = 5,
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cat-Head ONNX not found: {onnx_path}")

        # Create CPU-only ONNXRuntime session
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.classes_path = classes_path
        self.img_size = int(img_size)
        self.conf_raw = float(conf_raw)
        self.min_conf = float(min_conf)
        self.mid_conf = float(mid_conf)
        self.hi_count = float(hi_count)
        self.hi_priority = float(hi_priority)
        self.max_det = int(max_det)

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to BGR numpy array."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)[:, :, ::-1]
        return arr

    @staticmethod
    def _encode_jpeg_b64(bgr: np.ndarray) -> str:
        """Encode BGR image to base64 JPEG string."""
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _encode_jpeg_bytes(bgr: np.ndarray) -> bytes | None:
        """Encode BGR image to JPEG bytes."""
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return buf.tobytes() if ok else None

    def _pick_idx(
        self, confs: np.ndarray, xyxy: np.ndarray, mask: np.ndarray | list[int]
    ):
        idxs = (
            np.where(mask)[0]
            if isinstance(mask, np.ndarray)
            else np.array(mask, dtype=int)
        )
        if len(idxs) == 0:
            return None
        best_conf = confs[idxs].max()
        best_idxs = idxs[confs[idxs] == best_conf]
        if len(best_idxs) == 1:
            return int(best_idxs[0])
        areas = (xyxy[best_idxs, 2] - xyxy[best_idxs, 0]) * (
            xyxy[best_idxs, 3] - xyxy[best_idxs, 1]
        )
        return int(best_idxs[np.argmax(areas)])

    def _draw_preview(
        self,
        bgr: np.ndarray,
        xyxy: np.ndarray,
        confs: np.ndarray,
        chosen_idx: int | None,
        min_conf: float,
    ) -> bytes | None:
        """Draw bounding boxes on image and return as JPEG bytes."""
        canvas = bgr.copy()
        for i, (x1, y1, x2, y2) in enumerate(xyxy.astype(int)):
            conf = float(confs[i])
            if conf < min_conf:
                continue
            color = (0, 255, 0)
            thickness = 2
            if chosen_idx is not None and i == int(chosen_idx):
                color = (0, 0, 255)
                thickness = 3
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return self._encode_jpeg_bytes(canvas)

    def _run_onnx(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run ONNX detection model and return (confs, xyxy) arrays.

        The exact output format depends on the exported model. This helper assumes
        a common layout where the first output contains boxes+scores as
        [N, 6] = [x1, y1, x2, y2, score, class_id]. Adjust this parsing if your
        model differs.
        """
        img = bgr
        h, w = img.shape[:2]
        # Resize with letterbox/pad to self.img_size while keeping aspect ratio
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        inp = canvas.astype("float32") / 255.0
        inp = np.transpose(inp, (2, 0, 1))  # HWC -> CHW
        inp = np.expand_dims(inp, axis=0)

        outputs = self.session.run(None, {self.input_name: inp})
        preds = outputs[0]

        if preds.ndim == 3:
            preds = preds[0]

        boxes = []
        confs = []
        for det in preds:
            x1, y1, x2, y2, score, cls_id = det[:6]
            if score < self.conf_raw:
                continue

            # Map back to original image coordinates
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            boxes.append([x1, y1, x2, y2])
            confs.append(score)

        if not boxes:
            return np.array([]), np.empty((0, 4), dtype=int)

        confs_arr = np.array(confs, dtype=float)
        xyxy_arr = np.array(boxes, dtype=float)
        return confs_arr, xyxy_arr

    def detect(self, image_bytes: bytes, include_roi_b64: bool = True) -> FaceResult:
        """Detect cat faces and return structured result (CPU-only)."""
        bgr = self._bytes_to_bgr(image_bytes)

        confs, xyxy = self._run_onnx(bgr)

        if confs.size == 0 or xyxy.shape[0] == 0:
            preview = self._encode_jpeg_bytes(bgr)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note="No boxes",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        xyxy = xyxy.astype(int)

        keep_min = confs >= self.min_conf
        if not keep_min.any():
            preview = self._draw_preview(bgr, xyxy, confs, None, self.min_conf)
            return FaceResult(
                ok=False,
                faces_count=0,
                chosen_conf=None,
                box=None,
                note=f"No boxes >= {self.min_conf:.2f}",
                kept_confs_ge_min=[],
                meta={"img_size": self.img_size, "conf_raw": self.conf_raw},
                preview_jpeg=preview,
                roi_jpeg=None,
                roi_b64=None,
            )

        n_hi = int((confs >= self.hi_count).sum())
        faces_count = n_hi if n_hi >= 2 else 1

        idx = self._pick_idx(confs, xyxy, confs >= self.hi_priority)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.mid_conf)
        if idx is None:
            idx = self._pick_idx(confs, xyxy, confs >= self.min_conf)

        x1, y1, x2, y2 = map(int, xyxy[idx])
        roi_jpeg = None
        roi_b64 = None
        roi = bgr[y1:y2, x1:x2]
        ok_roi, buf_roi = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok_roi:
            roi_bytes = buf_roi.tobytes()
            roi_jpeg = roi_bytes
            if include_roi_b64:
                roi_b64 = base64.b64encode(roi_bytes).decode("ascii")

        preview = self._draw_preview(bgr, xyxy, confs, idx, self.min_conf)

        note = (
            "Single face detected"
            if faces_count == 1
            else f"Multiple faces detected ({faces_count})"
        )

        return FaceResult(
            ok=True,
            faces_count=faces_count,
            chosen_conf=float(confs[idx]),
            box=[x1, y1, x2, y2],
            note=note,
            kept_confs_ge_min=confs[keep_min].tolist(),
            meta={
                "img_size": self.img_size,
                "conf_raw": self.conf_raw,
                "min_conf": self.min_conf,
                "mid_conf": self.mid_conf,
                "hi_count": self.hi_count,
                "hi_priority": self.hi_priority,
            },
            preview_jpeg=preview,
            roi_jpeg=roi_jpeg,
            roi_b64=roi_b64,
        )
