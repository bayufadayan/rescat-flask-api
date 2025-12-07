import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from matplotlib import cm


class AreaClassifier:
    """
    Classifier for cat facial areas (eyes, ears, mouth) using EfficientNetB1 models.
    Also generates Grad-CAM visualizations for explainability.
    """
    
    def __init__(self, eye_model_path, ear_model_path, mouth_model_path):
        """Initialize the classifier with three EfficientNetB1 models."""
        print(f"Loading eye model from {eye_model_path}...")
        self.eye_model = keras.models.load_model(eye_model_path)
        
        print(f"Loading ear model from {ear_model_path}...")
        self.ear_model = keras.models.load_model(ear_model_path)
        
        print(f"Loading mouth model from {mouth_model_path}...")
        self.mouth_model = keras.models.load_model(mouth_model_path)
        
        # Model mapping
        self.model_map = {
            "right_eye": self.eye_model,
            "left_eye": self.eye_model,
            "mouth": self.mouth_model,
            "right_ear": self.ear_model,
            "left_ear": self.ear_model
        }
        
        print("All classification models loaded successfully!")
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for EfficientNetB1."""
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_resized = img.resize((240, 240))
        img_array = np.array(img_resized)
        img_input = preprocess_input(np.expand_dims(img_array, axis=0))
        
        return img_input, img_resized
    
    def classify_area(self, area_name, image_bytes):
        """Classify a facial area as normal or abnormal."""
        model = self.model_map[area_name]
        img_input, _ = self.preprocess_image(image_bytes)
        
        pred = model.predict(img_input, verbose=0)[0][0]
        
        if pred > 0.5:
            label = "normal"
            confidence = float(pred)
        else:
            label = "abnormal"
            confidence = float(1 - pred)
        
        return {
            "label": label,
            "confidence": confidence
        }
    
    def resolve_base_layer_name(self, model):
        """Find the base layer (efficientnetb1)."""
        for layer in model.layers:
            if 'efficientnetb1' in layer.name.lower():
                return layer.name
        
        for layer in model.layers:
            if 'efficient' in layer.name.lower():
                return layer.name
        
        raise ValueError("Could not find EfficientNetB1 base layer")
    
    def resolve_last_conv_name(self, model, base_layer_name):
        """Find last convolutional layer."""
        base_model = model.get_layer(base_layer_name)
        
        for layer in base_model.layers:
            if 'top_conv' in layer.name.lower():
                return layer.name
        
        conv_layers = [l.name for l in base_model.layers if 'conv' in l.name.lower()]
        if conv_layers:
            return conv_layers[-1]
        
        raise ValueError("Could not find last convolutional layer")
    
    def call_layer(self, layer, x, training=False):
        """Call a Keras layer while handling optional args and single-item lists."""
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        try:
            return layer(x, training=training)
        except TypeError:
            return layer(x)
    
    def build_forward_with_cam(self, model, base_layer_name, last_conv_name):
        """Replicate the working Grad-CAM forward pass from the reference notebook."""
        layers = model.layers[:]
        base_idx = None
        for idx, lyr in enumerate(layers):
            if lyr.name == base_layer_name:
                base_idx = idx
                break
        if base_idx is None:
            raise ValueError(f"Base layer '{base_layer_name}' not found in model.layers")
        pre_layers = layers[:base_idx]
        post_layers = layers[base_idx + 1:]
        base = model.get_layer(base_layer_name)
        conv_layer = base.get_layer(last_conv_name)
        base_multi = keras.Model(
            inputs=base.input,
            outputs=[conv_layer.output, base.output],
            name=f"{base_layer_name}_cam_extractor"
        )
        
        def forward(x):
            z = x
            for lyr in pre_layers:
                if isinstance(lyr, keras.layers.InputLayer):
                    continue
                z = self.call_layer(lyr, z, training=False)
            conv_out, z = base_multi(z, training=False)
            for lyr in post_layers:
                z = self.call_layer(lyr, z, training=False)
            return conv_out, z
        
        return forward
    
    def make_gradcam_heatmap(self, img_input, model, base_layer_name, last_conv_name):
        """Generate Grad-CAM heatmap using the forward pass builder."""
        forward = self.build_forward_with_cam(model, base_layer_name, last_conv_name)
        x = tf.convert_to_tensor(img_input, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            conv_out, pred = forward(x)
            tape.watch(conv_out)
            pred_tensor = pred
            if len(pred_tensor.shape) == 2 and pred_tensor.shape[-1] == 1:
                p = pred_tensor[:, 0]
            elif len(pred_tensor.shape) == 1:
                p = pred_tensor
            else:
                p = tf.reduce_mean(pred_tensor, axis=list(range(1, len(pred_tensor.shape))))
            loss = tf.where(p >= 0.5, p, 1.0 - p)
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            raise RuntimeError("Gradient computation failed")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv0 = conv_out[0]
        heatmap = tf.reduce_sum(conv0 * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.45):
        """Overlay heatmap on original image."""
        h, w = original_img.shape[:2]
        
        heat_resized = tf.image.resize(
            tf.expand_dims(heatmap, axis=-1),
            (h, w)
        ).numpy().squeeze()
        
        colored = cm.get_cmap("jet")(heat_resized)[..., :3]
        colored = (colored * 255).astype(np.uint8)
        
        out = original_img.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    
    def generate_gradcam(self, area_name, image_bytes):
        """Generate Grad-CAM visualization for a facial area."""
        model = self.model_map[area_name]
        img_input, img_resized = self.preprocess_image(image_bytes)
        
        base_layer_name = self.resolve_base_layer_name(model)
        last_conv_name = self.resolve_last_conv_name(model, base_layer_name)
        
        heatmap = self.make_gradcam_heatmap(img_input, model, base_layer_name, last_conv_name)
        
        overlayed_img = self.overlay_heatmap(heatmap, np.array(img_resized))
        
        output = io.BytesIO()
        Image.fromarray(overlayed_img).save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.getvalue()
