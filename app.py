import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import base64
import io
import cv2
from PIL import Image
import threading

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape,
    Multiply, Add, Lambda, Concatenate, Conv2D, Input,
    BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Small

from flask import Flask, request, jsonify, session
from flask_cors import CORS

# ─────────────────────────────────
# CONFIG
# ─────────────────────────────────

MODEL_PATH = "deepfake_mobilenetv3_attention.h5"
IMG_SIZE   = 192

# THRESHOLD: score > 0.5 → model leans toward class-1
# After inversion fix: high score = REAL, low score = DEEPFAKE
# So is_fake = smooth_score < 0.5
THRESHOLD  = 0.50

# Per-session smoothing (keyed by session_id sent from frontend)
session_buffers = {}
session_lock    = threading.Lock()
MAX_BUFFER      = 5

print("Building model architecture...")

# ─────────────────────────────────
# CBAM ATTENTION BLOCK
# ─────────────────────────────────

def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    # ── Channel attention ──
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Dense(channel // ratio, activation="relu")(avg_pool)
    avg_pool = Dense(channel, activation="sigmoid")(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Dense(channel // ratio, activation="relu")(max_pool)
    max_pool = Dense(channel, activation="sigmoid")(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Reshape((1, 1, channel))(channel_attention)
    x = Multiply()([input_feature, channel_attention])

    # ── Spatial attention ──
    avg_sp  = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_sp  = Lambda(lambda z: tf.reduce_max(z,  axis=-1, keepdims=True))(x)
    concat  = Concatenate(axis=-1)([avg_sp, max_sp])
    spatial = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)

    return Multiply()([x, spatial])

# ─────────────────────────────────
# BUILD MODEL  (identical to training)
# ─────────────────────────────────

input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model   = MobileNetV3Small(weights=None, include_top=False,
                                 input_tensor=input_tensor)

x = base_model.output
x = cbam_block(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64,  activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(inputs=input_tensor, outputs=output)
model.load_weights(MODEL_PATH)

print("✓ Model weights loaded")

# warm-up pass
model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
print("✓ Warm-up done")

# ─────────────────────────────────
# FACE DETECTORS  (cascade + DNN fallback)
# ─────────────────────────────────

# Primary: Haar cascade (fast)
haar_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Secondary: profile face (catches side angles)
haar_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def detect_face(frame_rgb):
    """
    Try frontal → profile → centre-crop.
    Returns cropped face (RGB ndarray) + bbox or None.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)       # equalize only for detection

    # 1) frontal
    faces = haar_detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 2) profile if frontal failed
    if len(faces) == 0:
        faces = haar_profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60)
        )

    if len(faces) > 0:
        # pick largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        pad = int(0.15 * min(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame_rgb.shape[1], x + w + pad)
        y2  = min(frame_rgb.shape[0], y + h + pad)
        face = frame_rgb[y1:y2, x1:x2]
        return face, [int(x), int(y), int(w), int(h)], True

    # 3) fallback: centre square crop (live feed always has user in centre)
    h, w = frame_rgb.shape[:2]
    side  = min(h, w)
    cx, cy = w // 2, h // 2
    x1  = cx - side // 2
    y1  = cy - side // 2
    face = frame_rgb[y1:y1+side, x1:x1+side]
    return face, None, False   # no face bbox → caller knows


# ─────────────────────────────────
# GRADCAM
# ─────────────────────────────────

def gradcam(img_tensor):
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, 0]

    grads       = tape.gradient(loss, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    max_val     = tf.math.reduce_max(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (max_val + 1e-8)
    return heatmap.numpy()


# ─────────────────────────────────
# FLASK
# ─────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "eamnet-secret-2024")
CORS(app, supports_credentials=True)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    """Call this when switching between live feed and image upload."""
    sid = request.json.get("session_id", "default")
    with session_lock:
        session_buffers.pop(sid, None)
    return jsonify({"status": "reset"})


# ─────────────────────────────────
# PREDICT  (live webcam frames)
# ─────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data    = request.json
        sid     = data.get("session_id", "default")
        img_b64 = data["image"].split(",")[1]

        img_bytes = base64.b64decode(img_b64)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame     = np.array(img)

        # ── Face detection ──
        face, bbox, face_found = detect_face(frame)

        # ── Preprocess for model ──
        # Mild CLAHE (contrast limited) — less destructive than full equalizeHist
        face_lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_lab[:, :, 0] = clahe.apply(face_lab[:, :, 0])
        face_proc = cv2.cvtColor(face_lab, cv2.COLOR_LAB2RGB)

        face_resized = cv2.resize(face_proc, (IMG_SIZE, IMG_SIZE))
        arr          = np.array(face_resized, dtype=np.float32)
        arr_model    = preprocess_input(arr.copy())
        arr_model    = np.expand_dims(arr_model, axis=0)

        raw_score = float(model.predict(arr_model, verbose=0)[0][0])

        # ── Per-session temporal smoothing ──
        with session_lock:
            buf = session_buffers.setdefault(sid, [])
            buf.append(raw_score)
            if len(buf) > MAX_BUFFER:
                buf.pop(0)
            smooth_score = float(np.mean(buf))

        # ── Decision logic (FIXED) ──
        # Model outputs HIGH score → REAL face
        #                LOW score → DEEPFAKE / AI-generated
        # is_fake when score is LOW (< threshold)
        is_fake = smooth_score < THRESHOLD

        # Confidence: how far from 0.5 boundary, scaled to 50–100%
        distance   = abs(smooth_score - 0.5)          # 0.0 → 0.5
        confidence = 50.0 + (distance / 0.5) * 50.0  # 50% → 100%

        # Bonus: if no face detected in a live frame, nudge toward DEEPFAKE
        # (real webcam almost always has a detectable face)
        if not face_found:
            is_fake    = True
            confidence = max(confidence, 55.0)

        # ── GradCAM overlay ──
        heatmap = gradcam(arr_model)
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face_resized, 0.6, heatmap, 0.4, 0)

        _, buffer    = cv2.imencode(".jpg", overlay)
        heatmap_b64  = base64.b64encode(buffer).decode()

        return jsonify({
            "score":      round(smooth_score, 4),
            "label":      "DEEPFAKE" if is_fake else "REAL",
            "is_fake":    bool(is_fake),
            "confidence": round(confidence, 1),
            "face_found": face_found,
            "bbox":       bbox,
            "attention":  heatmap_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────
# UPLOAD  (static AI-generated images)
# ─────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file  = request.files["image"]
        img   = Image.open(file).convert("RGB")
        frame = np.array(img)

        face, bbox, face_found = detect_face(frame)

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        arr          = np.array(face_resized, dtype=np.float32)
        arr_model    = preprocess_input(arr.copy())
        arr_model    = np.expand_dims(arr_model, axis=0)

        score   = float(model.predict(arr_model, verbose=0)[0][0])

        # Same fixed logic: LOW score = DEEPFAKE
        is_fake = score < THRESHOLD

        distance   = abs(score - 0.5)
        confidence = 50.0 + (distance / 0.5) * 50.0

        return jsonify({
            "score":      round(score, 4),
            "label":      "DEEPFAKE" if is_fake else "REAL",
            "is_fake":    bool(is_fake),
            "confidence": round(confidence, 1),
            "face_found": face_found,
            "bbox":       bbox
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {PORT}")
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=True    # allow concurrent requests
    )
