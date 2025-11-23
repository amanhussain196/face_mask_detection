import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Must be first command
st.set_page_config(page_title="Face Mask Detection")

st.title("😷 Face Mask Detection (ONNX Model)")
st.write("Upload an image to detect face mask status.")

MODEL_PATH = "mask_detector.onnx"
CASCADE_PATH = "face_detector/haarcascade_frontalface_default.xml"

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Load ONNX Model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

LABELS = ["Mask Incorrect", "With Mask", "Without Mask"]
COLORS = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("⚠ No face detected! Try a clearer photo.")
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            face_normalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = session.run(None, {input_name: face_input})[0][0]
            label_id = np.argmax(prediction)
            label = LABELS[label_id]
            color = COLORS[label_id]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(frame, caption="Result", use_column_width=True)
