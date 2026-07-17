# app.py — Streamlit deployment for Rock/Paper/Scissors gesture recognition
#
# Two input modes:
#   1. Camera snapshot (st.camera_input) — works on Streamlit Cloud / HF Spaces,
#      unlike a live cv2.VideoCapture loop which needs local hardware access.
#   2. Image upload — for testing with saved photos.
#
# Loads gesture_model.pkl, a full sklearn Pipeline (StandardScaler + classifier)
# saved by train.py, so no separate scaler file is needed.

import cv2
import numpy as np
import joblib
import streamlit as st
from PIL import Image
from landmarks import extract_landmarks, draw_landmarks

st.set_page_config(page_title="Rock Paper Scissors", page_icon="✊", layout="centered")


@st.cache_resource
def load_model(path="gesture_model.pkl"):
    return joblib.load(path)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def predict(model, image_bgr: np.ndarray):
    feat = extract_landmarks(image_bgr)
    if feat is None:
        return None, None, None
    pred = model.predict([feat])[0]
    proba = model.predict_proba([feat])[0]
    classes = model.classes_
    return pred, proba, classes


GESTURE_EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}


def main():
    st.title("✊ ✋ ✌️ Rock Paper Scissors Classifier")
    st.caption(
        "MediaPipe hand landmarks (21 keypoints, normalized) → Logistic Regression"
    )

    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "gesture_model.pkl not found. Run `python train.py` first, "
            "then place gesture_model.pkl next to app.py."
        )
        st.stop()

    tab1, tab2 = st.tabs(["📷 Camera", "📤 Upload Image"])

    image_bgr = None

    with tab1:
        st.write("Take a snapshot of your hand gesture.")
        cam_image = st.camera_input("Camera", label_visibility="collapsed")
        if cam_image is not None:
            image_bgr = pil_to_bgr(Image.open(cam_image))

    with tab2:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bgr = pil_to_bgr(Image.open(uploaded))

    if image_bgr is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input")
            annotated = draw_landmarks(image_bgr)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Prediction")
            pred, proba, classes = predict(model, image_bgr)

            if pred is None:
                st.warning("No hand detected. Try better lighting or move your hand into frame.")
            else:
                emoji = GESTURE_EMOJI.get(pred, "")
                st.markdown(f"## {emoji} **{pred.upper()}**")
                st.metric("Confidence", f"{proba.max():.1%}")

                st.write("Class probabilities:")
                for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                    st.progress(float(p), text=f"{GESTURE_EMOJI.get(cls,'')} {cls}: {p:.1%}")

    with st.expander("How it works"):
        st.markdown(
            "1. MediaPipe Hands detects 21 hand landmarks.\n"
            "2. Landmarks are centered on the wrist and scaled by hand size, "
            "so the features don't depend on hand position or distance from the camera.\n"
            "3. A Logistic Regression model (chosen after comparing 5 algorithms — "
            "see `compare_models.py`) classifies the normalized landmarks."
        )


if __name__ == "__main__":
    main()
