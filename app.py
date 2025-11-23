import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json
import os

st.set_page_config(page_title="Face Mask Detector 😷", layout="centered")

# ---------- CACHED LOADERS ----------

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("mask_detector_mobilenetv2.h5")
    with open("label_map.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    friendly_labels = meta["friendly_labels"]
    return model, class_names, friendly_labels


@st.cache_resource
def load_face_detector():
    cascade_path = os.path.join("face_detector", "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(
            f"Could not find Haar cascade at {cascade_path}. "
            "Make sure the file is in face_detector/ folder."
        )
    return cv2.CascadeClassifier(cascade_path)


# ---------- UTILS ----------

def preprocess_face(face_bgr, target_size=(224, 224)):
    face_resized = cv2.resize(face_bgr, target_size)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_rgb = face_rgb.astype("float32") / 255.0
    return np.expand_dims(face_rgb, axis=0)


def predict_faces(image_rgb, model, face_cascade, class_names, friendly_labels):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    detections = []
    COLORS = {
        "with_mask": (0, 255, 0),              # green
        "without_mask": (0, 0, 255),           # red
        "mask_weared_incorrect": (0, 255, 255) # yellow
    }

    for (x, y, w, h) in faces:
        face_roi = image_bgr[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        inp = preprocess_face(face_roi)
        preds = model.predict(inp, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        prob = float(preds[class_idx])

        class_name = class_names[class_idx]
        friendly = friendly_labels.get(class_name, class_name)
        color = COLORS.get(class_name, (255, 255, 255))

        # draw box + label
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        label_text = f"{friendly} ({prob:.2f})"
        cv2.putText(
            image_bgr,
            label_text,
            (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        detections.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "raw_label": class_name,
                "label": friendly,
                "confidence": prob,
            }
        )

    # back to RGB for display
    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb, detections


# ---------- UI ----------

st.title("Face Mask Detection 😷")
st.write(
    "Upload an image or take a photo. "
    "The app will detect faces and classify them as:\n"
    "- 😷 Wearing Mask\n"
    "- ❌ Not Wearing Mask\n"
    "- ⚠️ Incorrect Mask Position"
)

model, class_names, friendly_labels = load_model_and_labels()
face_cascade = load_face_detector()

mode = st.radio("Choose input source:", ["Upload image", "Use camera"])

uploaded_file = None

if mode == "Upload image":
    uploaded_file = st.file_uploader(
        "Upload a photo with faces", type=["jpg", "jpeg", "png"]
    )
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        uploaded_file = camera_image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if st.button("Run detection"):
        with st.spinner("Analyzing image..."):
            result_img, detections = predict_faces(
                image_np, model, face_cascade, class_names, friendly_labels
            )

        st.image(result_img, caption="Detection result", use_column_width=True)

        if not detections:
            st.info("No faces detected. Try another image.")
        else:
            st.subheader("Detections")
            for i, det in enumerate(detections, start=1):
                st.write(
                    f"**Face {i}:** {det['label']} — "
                    f"confidence: `{det['confidence']:.3f}`"
                )
else:
    st.info("Please upload an image or use the camera to start.")
