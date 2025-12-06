"""Cat image classifier using MobileNetV3 ONNX model."""

import io
import time
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOMESTIC_CAT_LABELS = {"tabby", "tiger cat", "Egyptian cat", "Persian cat", "Siamese cat"}

def load_imagenet_classes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def preprocess_image_bytes(image_bytes: bytes, size: Tuple[int,int]=(224,224)) -> np.ndarray:
    """Decode image bytes to normalized NCHW tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr

def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

@dataclass
class InferenceResult:
    ok: bool
    label: str
    cat_prob: float
    threshold: float
    topk: List[Dict[str, float]]
    meta: Dict[str, str]

class CatClassifier:
    """MobileNetV3-based cat image classifier."""
    
    def __init__(
        self,
        onnx_path: str = "models/validation_model/mobilenetv3_small.onnx",
        classes_path: str = "models/validation_model/imagenet_classes.txt",
        threshold: float = 0.50,
        topk: int = 3,
        providers: List[str] = None
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")

        self.classes = load_imagenet_classes(classes_path)
        self.threshold = float(threshold)
        self.topk = int(topk)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_name = os.path.basename(onnx_path)

    def predict(self, image_bytes: bytes) -> InferenceResult:
        """Classify image and return CAT/NON-CAT result."""
        t0 = time.time()
        x = preprocess_image_bytes(image_bytes)

        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = softmax(logits)[0]

        idxs = probs.argsort()[-self.topk:][::-1]
        topk_list = [{"label": self.classes[i], "prob": float(probs[i])} for i in idxs]

        cat_prob = 0.0
        for lbl in DOMESTIC_CAT_LABELS:
            if lbl in self.classes:
                cat_prob += float(probs[self.classes.index(lbl)])

        label = "CAT" if cat_prob >= self.threshold else "NON-CAT"
        latency_ms = int((time.time() - t0) * 1000)

        meta = {
            "model": self.model_name,
            "runtime": "onnxruntime-cpu",
            "latency_ms": str(latency_ms),
        }
        return InferenceResult(
            ok=True,
            label=label,
            cat_prob=round(cat_prob, 6),
            threshold=self.threshold,
            topk=topk_list,
            meta=meta,
        )

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python inference.py <path/to/image.jpg>")
        sys.exit(1)

    image_arg = sys.argv[1]
    base_dir = Path(__file__).resolve().parent

    p = Path(image_arg)
    if not p.exists():
        p = (base_dir / image_arg.lstrip(r"\/")).resolve()

    print(f"[cwd] {Path.cwd()}")
    print(f"[try] {p}")

    if not p.exists():
        print("ERROR: image file not found. Coba salah satu:")
        print(f" - python inference.py static{os.sep}images{os.sep}image-4.jpg")
        print(f" - python inference.py \"{(base_dir / 'static/images/image-4.jpg').resolve()}\"")
        sys.exit(1)

    with open(p, "rb") as f:
        image_bytes = f.read()

    clf = CatClassifier(
        onnx_path="models/validation_model/mobilenetv3_small.onnx",
        classes_path="models/validation_model/imagenet_classes.txt",
        threshold=float(os.getenv("THRESHOLD", 0.50)),
        topk=int(os.getenv("TOPK", 3)),
    )
    res = clf.predict(image_bytes)
    print({
        "ok": res.ok,
        "label": res.label,
        "cat_prob": res.cat_prob,
        "threshold": res.threshold,
        "topk": res.topk,
        "meta": res.meta,
    })
