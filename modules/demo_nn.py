import streamlit as st
import tensorflow as tf
import numpy as np
import random
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

sample_images = []
if class_labels:
    for label in class_labels:
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_dir):
            images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith((".jpg", ".png", ".jpeg"))]
            if images:
                sample_images.extend([(label, img) for img in images])  # เก็บรูปทั้งหมดของแต่ละ class

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def show():
    st.title("🧚‍♀️ Princess Classifier - Neural Network")
    
    if sample_images:
        # สุ่มรูปเจ้าหญิง Disney ทุกครั้งที่กดปุ่ม
        selected_label, selected_image_path = random.choice(sample_images)
        
        img = Image.open(selected_image_path)
        st.image(img, caption=f"📸 รูปตัวอย่าง: {selected_label}", use_container_width=True)
        
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        
        if class_labels:
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.write(f"✨ เจ้าหญิงที่ทำนายได้: **{predicted_class}**")
            st.write(f"🎯 ความมั่นใจ: **{confidence:.2f}%**")
        else:
            st.error("🚨 ไม่พบข้อมูล class labels! กรุณาตรวจสอบโฟลเดอร์ `datasources/princess`")
        
        # ปุ่มที่ทำนายรูปใหม่
        st.button("🔀 ทำนายรูปเจ้าหญิง Disney ใหม่", on_click=show)

# เรียกฟังก์ชันแสดงผล
show()
