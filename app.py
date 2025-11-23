import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

st.set_page_config(page_title="Face Mask Detection")
st.title("😷 Face Mask Detection (ONNX Model)")
st.write("Upload an image to detect mask status.")

# Load ONNX model
session = ort.InferenceSession("mask_detector.onnx")
input_name = session.get_inputs()[0].name

# Load DNN Face Detector
prototxt = "face_detector/deploy.prototxt.txt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxt, weights)

labels = ["Mask Incorrect", "With Mask", "Without Mask"]
colors = [(0,255,255), (0,255,0), (0,0,255)]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    results_text = []
    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.3:  # Lower threshold
            face_count += 1
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1, y1, x2, y2 = box.astype("int")

            # Add padding to include mask region
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_img = cv2.resize(face, (224,224))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            preds = session.run(None, {input_name: face_img})[0][0]
            idx = int(np.argmax(preds))
            conf = preds[idx] * 100

            label = f"{labels[idx]} ({conf:.1f}%)"
            color = colors[idx]

            results_text.append(f"Face {face_count}: **{label}**")

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, labels[idx], (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.subheader("Prediction Results")

    if not results_text:
        st.warning("⚠ No face detected. Try a clearer image.")
    else:
        for r in results_text:
            st.markdown(r)

    st.image(frame, caption="Detection Result", use_column_width=True)

