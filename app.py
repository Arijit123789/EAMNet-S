import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import base64
import io
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from flask import Flask, request, jsonify
from flask_cors import CORS


# ─────────────────────────────────
# CONFIG
# ─────────────────────────────────

MODEL_PATH = "deepfake_mobilenetv3_attention.h5"
IMG_SIZE   = 192
THRESHOLD  = 0.5        # neutral starting point — tune after checking DEBUG logs
BUFFER_MAX = 5          # temporal smoothing window

# global smoothing buffer (reset on no-face frames)
score_buffer = []

print("Building model...")


# ─────────────────────────────────
# CBAM ATTENTION BLOCK
# ─────────────────────────────────

def cbam_block(input_feature, ratio=8):

    channel = input_feature.shape[-1]

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Dense(channel // ratio, activation="relu")(avg_pool)
    avg_pool = Dense(channel, activation="sigmoid")(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Dense(channel // ratio, activation="relu")(max_pool)
    max_pool = Dense(channel, activation="sigmoid")(max_pool)

    channel_attention = Add()([avg_pool, max_pool])
    channel_attention = Reshape((1, 1, channel))(channel_attention)

    x = Multiply()([input_feature, channel_attention])

    avg_pool_s = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool_s = Lambda(lambda z: tf.reduce_max(z,  axis=-1, keepdims=True))(x)

    concat = Concatenate(axis=-1)([avg_pool_s, max_pool_s])

    spatial_attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    return Multiply()([x, spatial_attention])


# ─────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────

from tensorflow.keras.applications import MobileNetV3Small

input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = MobileNetV3Small(
    weights=None,
    include_top=False,
    input_tensor=input_tensor
)

x = base_model.output
x = cbam_block(x)

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(inputs=input_tensor, outputs=output)

# ── load weights safely ──────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if os.path.getsize(MODEL_PATH) < 1024:
    raise ValueError(f"Model file looks corrupt / empty: {MODEL_PATH}")

model.load_weights(MODEL_PATH)
print("Model loaded successfully")

# ── warm-up pass ─────────────────────────────────────────────────────────────
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
warmup_score = float(model.predict(dummy, verbose=0)[0][0])
print(f"[WARMUP] dummy score = {warmup_score:.4f}  (should be near 0.5 for untrained / random)")


# ─────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────
# GRADCAM HEATMAP
# ─────────────────────────────────

def gradcam(img_tensor):
    """Return a (IMG_SIZE × IMG_SIZE) float32 heatmap, or None on failure."""
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                last_conv = layer.name
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        pooled_grads  = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out_0    = conv_outputs[0]
        heatmap       = conv_out_0 @ pooled_grads[..., tf.newaxis]
        heatmap       = tf.squeeze(heatmap)
        max_val       = tf.math.reduce_max(heatmap)

        if max_val == 0:
            return None

        heatmap = tf.maximum(heatmap, 0) / max_val
        return heatmap.numpy()

    except Exception as e:
        print(f"[GradCAM ERROR] {e}")
        return None


def build_overlay(face_rgb, heatmap_raw):
    """Overlay GradCAM heatmap on face image; return base64 JPEG string or None."""
    try:
        heatmap = cv2.resize(heatmap_raw, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face_rgb, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode(".jpg", overlay)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        print(f"[OVERLAY ERROR] {e}")
        return None


# ─────────────────────────────────
# HELPERS
# ─────────────────────────────────

def preprocess_face(face_rgb):
    """Lighting normalisation + resize + model preprocessing."""
    # histogram equalisation on luminance channel only
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    face_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))
    arr = np.array(face_resized, dtype=np.float32)

    arr_model = preprocess_input(arr.copy())
    arr_model = np.expand_dims(arr_model, axis=0)

    return face_resized, arr_model


def is_blank_frame(frame_rgb, std_threshold=8.0):
    """Return True if the frame is nearly uniform (covered camera, etc.)."""
    return float(np.std(frame_rgb)) < std_threshold


# ─────────────────────────────────
# FLASK
# ─────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


# ─────────────────────────────────
# PREDICT API  (webcam stream)
# ─────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():

    global score_buffer

    try:
        data    = request.json
        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame   = np.array(img)

        # ── blank / covered camera guard ─────────────────────────────────────
        if is_blank_frame(frame):
            score_buffer.clear()
            return jsonify({
                "score":      0.0,
                "label":      "NO SIGNAL",
                "is_fake":    False,
                "confidence": 0.0,
                "bbox":       None,
                "attention":  None,
                "note":       "Frame appears blank or camera is covered."
            })

        # ── face detection ───────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            # no face — clear buffer so we don't contaminate future frames
            score_buffer.clear()
            return jsonify({
                "score":      0.0,
                "label":      "NO FACE",
                "is_fake":    False,
                "confidence": 0.0,
                "bbox":       None,
                "attention":  None,
                "note":       "No face detected in frame."
            })

        # use the largest detected face
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h   = faces_sorted[0]
        pad = int(0.2 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        bbox      = [int(x), int(y), int(w), int(h)]

        # ── preprocessing ────────────────────────────────────────────────────
        face_resized, arr_model = preprocess_face(face_crop)

        # ── model inference ──────────────────────────────────────────────────
        raw_score = float(model.predict(arr_model, verbose=0)[0][0])

        # guard against NaN / Inf (bad weights or preprocessing)
        if not np.isfinite(raw_score):
            print(f"[WARN] Non-finite score: {raw_score} — skipping frame")
            return jsonify({"error": "Model returned non-finite score. Check weights."}), 500

        print(f"[DEBUG] raw_score={raw_score:.4f}  faces={len(faces)}")

        # ── temporal smoothing ───────────────────────────────────────────────
        score_buffer.append(raw_score)
        if len(score_buffer) > BUFFER_MAX:
            score_buffer.pop(0)
        smooth_score = float(np.mean(score_buffer))

        # ── classification ───────────────────────────────────────────────────
        #  MODEL CONVENTION: output > THRESHOLD  →  FAKE
        #  If everything is showing as FAKE, flip to:  smooth_score < THRESHOLD
        #  Check [DEBUG] logs — if real faces score > 0.65, flip the sign below.
        is_fake    = smooth_score > THRESHOLD
        confidence = smooth_score if is_fake else (1.0 - smooth_score)

        # ── GradCAM (non-fatal) ──────────────────────────────────────────────
        heatmap_raw = gradcam(arr_model)
        attention_b64 = build_overlay(face_resized, heatmap_raw) if heatmap_raw is not None else None

        return jsonify({
            "score":      round(smooth_score, 4),
            "raw_score":  round(raw_score,    4),   # useful for debugging
            "label":      "DEEPFAKE" if is_fake else "REAL",
            "is_fake":    bool(is_fake),
            "confidence": round(confidence * 100, 1),
            "bbox":       bbox,
            "attention":  attention_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────
# IMAGE UPLOAD API
# ─────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():

    try:
        file = request.files["image"]
        img  = Image.open(file).convert("RGB")
        frame = np.array(img)

        if is_blank_frame(frame):
            return jsonify({"error": "Uploaded image appears blank."}), 400

        gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h   = faces_sorted[0]
            pad = int(0.2 * w)
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(frame.shape[1], x + w + pad)
            y2  = min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame   # fall back to full image if no face found

        face_resized, arr_model = preprocess_face(roi)

        score = float(model.predict(arr_model, verbose=0)[0][0])

        if not np.isfinite(score):
            return jsonify({"error": "Model returned non-finite score."}), 500

        print(f"[DEBUG/upload] score={score:.4f}")

        is_fake    = score > THRESHOLD
        confidence = score if is_fake else (1.0 - score)

        heatmap_raw   = gradcam(arr_model)
        attention_b64 = build_overlay(face_resized, heatmap_raw) if heatmap_raw is not None else None

        return jsonify({
            "score":      round(score, 4),
            "label":      "DEEPFAKE" if is_fake else "REAL",
            "is_fake":    bool(is_fake),
            "confidence": round(confidence * 100, 1),
            "attention":  attention_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────
# RESET BUFFER  (optional utility)
# ─────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset_buffer():
    global score_buffer
    score_buffer.clear()
    return jsonify({"status": "buffer cleared"})


# ─────────────────────────────────

if __name__ == "__main__":
    print("Server started on http://0.0.0.0:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=False
    )
