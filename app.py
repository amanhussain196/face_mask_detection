import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

st.set_page_config(page_title="Face Mask Detection")
st.title("😷 Face Mask Detection")

model_path = "mask_detector.onnx"
face_path = "face_detector"

# Load ONNX model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Load face detector
prototxt = f"{face_path}/deploy.prototxt"
weights = f"{face_path}/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt, weights)

labels = ["Mask Incorrect", "With Mask", "Without Mask"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1, y1, x2, y2 = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue

            face_img = cv2.resize(face, (224,224))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            preds = session.run(None, {input_name: face_img})[0][0]
            label_idx = np.argmax(preds)
            label = labels[label_idx]
            color = (0,255,0) if label_idx == 1 else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.image(frame, caption="Result", use_column_width=True)
