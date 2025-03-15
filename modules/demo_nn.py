import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

MODEL_PATH = "disney_princess_resnet50.h5"

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"🚨 โหลดโมเดลไม่สำเร็จ: {e}")
        return None

model = load_model()

DATA_DIR = "datasources/princess"
class_labels = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def show():  # ✅ เพิ่มฟังก์ชัน show()
    st.title("🧚‍♀️ Princess Classifier - Neural Network")
    
    uploaded_file = st.file_uploader("📤 อัปโหลดรูปเจ้าหญิง Disney", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and model:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 รูปที่อัปโหลด", use_column_width=True)
        
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        
        if class_labels:
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.write(f"✨ เจ้าหญิงที่ทำนายได้: **{predicted_class}**")
            st.write(f"🎯 ความมั่นใจ: **{confidence:.2f}%**")
        else:
            st.error("🚨 ไม่พบข้อมูล class labels! กรุณาตรวจสอบโฟลเดอร์ `datasources/princess`")
