import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# MUST be first Streamlit call
st.set_page_config(page_title="Face Mask Detection")

st.title("😷 Face Mask Detection (ONNX Model)")
st.write("Upload an image to detect face mask status.")

# Load ONNX model
session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels = ["Mask Incorrect", "With Mask", "Without Mask"]
colors = [(0,255,255), (0,255,0), (0,0,255)] # Yellow / Green / Red

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("⚠ No face detected! Try using a clearer photo.")
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224,224))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            preds = session.run(None, {input_name: face_resized})[0][0]
            label_idx = int(np.argmax(preds))
            label = labels[label_idx]
            color = colors[label_idx]

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(frame, caption="Result", use_column_width=True)
