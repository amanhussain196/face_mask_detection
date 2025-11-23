import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Face Mask Detection")
st.title("😷 Face Mask Detection")

# Load mask detection model
model = load_model("mask_detector_mobilenetv2.h5")

# Load Haar face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = ["Mask", "No Mask"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face, (224, 224))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img)[0]
        label_index = np.argmax(preds)
        label = labels[label_index]
        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    st.image(frame, caption="Detection Result", use_column_width=True)
