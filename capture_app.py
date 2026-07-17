# capture_app.py — Streamlit tool for capturing labeled rock/paper/scissors
# photos and saving them straight into the same folder structure train.py
# already expects.
#
# Run with:
#   streamlit run capture_app.py
#
# Every photo you save goes into:
#   rockpaperscissors/rock/       rockpaperscissors/paper/      rockpaperscissors/scissors/
# right alongside the Kaggle dataset images, so the next time you run
# train.py, your own webcam shots (different angles, distances, lighting)
# get folded into training automatically. No manual file moving needed.
#
# Tip: capture 20-30 photos per class, deliberately varying:
#   - distance from camera (close and far)
#   - angle/tilt (hand raised near face, hand tilted sideways, etc.)
#   - rotation (turned slightly left/right, not just flat-on)
#   - lighting (near a window, under overhead light, etc.)
# That variety is what actually teaches the model to generalize.

import os
import time

import numpy as np
import streamlit as st
from PIL import Image

from landmarks import extract_landmarks, draw_landmarks

DATA_DIR = "rockpaperscissors"
CLASSES = ["rock", "paper", "scissors"]
GESTURE_EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}

st.set_page_config(page_title="RPS Photo Capture", page_icon="📸", layout="centered")


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    import cv2
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ensure_folders():
    for c in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)


def count_images(label: str) -> int:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def save_image(pil_image: Image.Image, label: str) -> str:
    folder = os.path.join(DATA_DIR, label)
    os.makedirs(folder, exist_ok=True)
    fname = f"custom_{int(time.time() * 1000)}.jpg"
    path = os.path.join(folder, fname)
    pil_image.convert("RGB").save(path, "JPEG", quality=95)
    return path


def main():
    ensure_folders()

    st.title("📸 RPS Training Photo Capture")
    st.caption(
        "Snap a photo, confirm a hand is detected, pick the correct label, and save it "
        "straight into your training folder. Vary angle, distance, and lighting for best results."
    )

    counts = {c: count_images(c) for c in CLASSES}
    cols = st.columns(3)
    for col, c in zip(cols, CLASSES):
        col.metric(f"{GESTURE_EMOJI[c]} {c}", counts[c])

    st.divider()

    cam_image = st.camera_input("Take a photo")

    if cam_image is not None:
        pil_image = Image.open(cam_image)
        image_bgr = pil_to_bgr(pil_image)

        feat = extract_landmarks(image_bgr)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Preview")
            if feat is not None:
                import cv2
                annotated = draw_landmarks(image_bgr)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.success("Hand detected — landmarks look good.")
            else:
                st.image(pil_image, use_container_width=True)
                st.warning(
                    "No hand detected in this shot. You can still save it, but it won't "
                    "be usable for training since train.py skips images with no detected hand. "
                    "Try adjusting lighting or hand position and retake."
                )

        with col2:
            st.subheader("Label & Save")
            label = st.radio(
                "What gesture is this?",
                CLASSES,
                format_func=lambda c: f"{GESTURE_EMOJI[c]} {c}",
                horizontal=False,
            )

            disabled = feat is None
            if st.button("💾 Save to training set", disabled=disabled, use_container_width=True):
                path = save_image(pil_image, label)
                st.success(f"Saved to `{path}`")
                st.rerun()

            if disabled:
                st.caption("Save is disabled until a hand is detected in the photo.")

    st.divider()
    with st.expander("What happens next"):
        st.markdown(
            "1. Capture 20–30 photos per class, varying angle, distance, rotation, and lighting.\n"
            "2. Once you have a good spread, delete the old `gesture_model.pkl` "
            "(the feature space includes your new photos now, so the old model is stale).\n"
            "3. Run `python train.py` again — it will automatically pick up every image "
            "in `rockpaperscissors/rock/`, `/paper/`, and `/scissors/`, including these new ones.\n"
            "4. Redeploy the new `gesture_model.pkl` with your Streamlit app."
        )


if __name__ == "__main__":
    main()
