import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageDraw


@dataclass
class LandmarkResult:
    right_eye: bytes
    left_eye: bytes
    mouth: bytes
    right_ear: bytes
    left_ear: bytes
    landmarked_image: bytes


class LandmarkDetector:
    IMG_SIZE = 224
    MIN_RES = 512
    MAX_RES = 1400

    def __init__(self, bbox_onnx_path: str, landmark_onnx_path: str):
        self.bbox_session = ort.InferenceSession(bbox_onnx_path)
        self.landmark_session = ort.InferenceSession(landmark_onnx_path)
        self.bbox_input_name = self.bbox_session.get_inputs()[0].name
        self.landmark_input_name = self.landmark_session.get_inputs()[0].name

    def validate_and_resize_image(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if max(w, h) < self.MIN_RES:
            raise ValueError(
                f"Resolution too small ({w}x{h}). Minimum {self.MIN_RES}px on longest side."
            )
        if max(w, h) > self.MAX_RES:
            scale = self.MAX_RES / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def resize_with_padding(self, img: Image.Image) -> Tuple[Image.Image, int, int, float]:
        w, h = img.size
        scale = self.IMG_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (self.IMG_SIZE, self.IMG_SIZE))
        pad_x = (self.IMG_SIZE - new_w) // 2
        pad_y = (self.IMG_SIZE - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))
        return canvas, pad_x, pad_y, scale

    def preprocess(self, x: Image.Image) -> np.ndarray:
        x = np.array(x).astype("float32")
        x = x / 127.5 - 1.0
        return np.expand_dims(x, axis=0)

    def predict_bbox(self, img: Image.Image) -> Tuple[int, int, int, int]:
        img224, pad_x, pad_y, scale = self.resize_with_padding(img)
        x = self.preprocess(img224)
        bbox_pred = self.bbox_session.run(None, {self.bbox_input_name: x})[0][0][:4]
        x0 = (bbox_pred[0] - pad_x) / scale
        y0 = (bbox_pred[1] - pad_y) / scale
        x1 = (bbox_pred[2] - pad_x) / scale
        y1 = (bbox_pred[3] - pad_y) / scale
        return int(x0), int(y0), int(x1), int(y1)

    def predict_landmarks(self, img: Image.Image, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        roi = img.crop(bbox)
        roi224, roi_px, roi_py, roi_scale = self.resize_with_padding(roi)
        x_roi = self.preprocess(roi224)
        lm_pred = self.landmark_session.run(None, {self.landmark_input_name: x_roi})[0][0]
        lm = lm_pred.reshape(-1, 2)
        lm[:, 0] = ((lm[:, 0] - roi_px) / roi_scale) + bbox[0]
        lm[:, 1] = ((lm[:, 1] - roi_py) / roi_scale) + bbox[1]
        return lm

    def determine_scenario(self, img: Image.Image) -> int:
        w, h = img.size
        longest = max(w, h)
        if longest < 1100:
            return 1
        elif longest <= 1400:
            return 2
        else:
            return 3

    def crop_point_scaled(self, img: Image.Image, p: np.ndarray, size: int = 128, scale: float = 2.2) -> Image.Image:
        x, y = p
        half = (size * scale) / 2
        return img.crop((int(x - half), int(y - half), int(x + half), int(y + half)))

    def crop_ear_square(
        self,
        img: Image.Image,
        p: np.ndarray,
        size: int = 200,
        top_ratio: float = 0.2,
        extend_ratio: float = 0.6,
        shift_x: int = 0,
    ) -> Image.Image:
        x, y = p
        img_w, img_h = img.size
        extended_size = size + (size * extend_ratio)
        top = y - (size * top_ratio)
        bottom = top + extended_size
        if top < 0:
            top = 0
            bottom = extended_size
        if bottom > img_h:
            bottom = img_h
            top = bottom - extended_size
        left = x - (extended_size / 2) + shift_x
        right = left + extended_size
        if left < 0:
            left = 0
            right = extended_size
        if right > img_w:
            right = img_w
            left = right - extended_size
        return img.crop((int(left), int(top), int(right), int(bottom)))

    def crop_landmarks(self, img: Image.Image, landmarks: np.ndarray) -> LandmarkResult:
        scenario = self.determine_scenario(img)
        if scenario == 1:
            right_eye = self.crop_point_scaled(img, landmarks[0], size=60, scale=2.2)
            left_eye = self.crop_point_scaled(img, landmarks[1], size=60, scale=2.2)
            mouth = self.crop_point_scaled(img, landmarks[2], size=80, scale=2.4)
            right_ear = self.crop_ear_square(img, landmarks[3], size=105, top_ratio=0.18, extend_ratio=1.1, shift_x=70)
            left_ear = self.crop_ear_square(img, landmarks[4], size=105, top_ratio=0.18, extend_ratio=1.1, shift_x=-70)
        elif scenario == 2:
            right_eye = self.crop_point_scaled(img, landmarks[0], size=78, scale=2.2)
            left_eye = self.crop_point_scaled(img, landmarks[1], size=78, scale=2.2)
            mouth = self.crop_point_scaled(img, landmarks[2], size=98, scale=2.4)
            right_ear = self.crop_ear_square(img, landmarks[3], size=123, top_ratio=0.18, extend_ratio=1.1, shift_x=70)
            left_ear = self.crop_ear_square(img, landmarks[4], size=123, top_ratio=0.18, extend_ratio=1.1, shift_x=-70)
        else:
            right_eye = self.crop_point_scaled(img, landmarks[0], size=85, scale=2.2)
            left_eye = self.crop_point_scaled(img, landmarks[1], size=85, scale=2.2)
            mouth = self.crop_point_scaled(img, landmarks[2], size=105, scale=2.4)
            right_ear = self.crop_ear_square(img, landmarks[3], size=134, top_ratio=0.18, extend_ratio=1.1, shift_x=70)
            left_ear = self.crop_ear_square(img, landmarks[4], size=134, top_ratio=0.18, extend_ratio=1.1, shift_x=-70)

        def to_bytes(crop: Image.Image) -> bytes:
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)
            return buf.getvalue()

        landmarked_image = self.draw_landmarks(img, landmarks)
        
        return LandmarkResult(
            right_eye=to_bytes(right_eye),
            left_eye=to_bytes(left_eye),
            mouth=to_bytes(mouth),
            right_ear=to_bytes(right_ear),
            left_ear=to_bytes(left_ear),
            landmarked_image=to_bytes(landmarked_image),
        )

    def draw_landmarks(self, img: Image.Image, landmarks: np.ndarray) -> Image.Image:
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        for i, (x, y) in enumerate(landmarks):
            r = 8
            draw.ellipse([x - r, y - r, x + r, y + r], fill="yellow", outline="orange", width=2)
        
        connections = [
            (3, 0),
            (0, 1),
            (1, 4),
            (4, 2),
            (2, 3),
        ]
        
        for start, end in connections:
            x1, y1 = landmarks[start]
            x2, y2 = landmarks[end]
            draw.line([x1, y1, x2, y2], fill="yellow", width=3)
        
        return img_copy

    def detect_and_crop(self, img: Image.Image) -> LandmarkResult:
        img = self.validate_and_resize_image(img)
        bbox = self.predict_bbox(img)
        landmarks = self.predict_landmarks(img, bbox)
        return self.crop_landmarks(img, landmarks)

    def detect_and_crop_from_url(self, url: str, timeout: int = 10) -> LandmarkResult:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return self.detect_and_crop(img)
