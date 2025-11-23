import streamlit as st
st.set_page_config(page_title="Face Mask Detection")  # must be first Streamlit UI call

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

st.title("😷 Face Mask Detection (ONNX Model)")

# Load ONNX model
session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name

# Load Haar face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = ["Mask Incorrect", "With Mask", "Without Mask"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image and convert RGBA -> RGB if transparent
    image = Image.open(uploaded_file).convert("RGB")
    
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("⚠️ No face detected! Try using a clearer photo.")
        st.image(frame, caption="Uploaded Image", use_column_width=True)
        st.stop()

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        # ONNX preprocessing
        face_img = cv2.resize(face, (224, 224))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = session.run(None, {input_name: face_img})[0][0]
        idx = np.argmax(preds)

        label = labels[idx]
        color = (0, 255, 0) if idx == 1 else (0, 0, 255)
        confidence = preds[idx] * 100
        text = f"{label} {confidence:.1f}%"

        # bounding box & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.image(frame, caption="Detection Result", use_column_width=True)
else:
    st.info("👆 Upload an image to begin detection!")
