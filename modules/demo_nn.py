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
            images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith((".jpg", ".jpeg", ".png"))]
            if images:
                sample_images.extend([(label, img) for img in images])  # เก็บรูปทั้งหมดของแต่ละ class

def preprocess_image(img):
    # ตรวจสอบขนาดรูปภาพ
    if img.size != (224, 224):
        img = img.resize((224, 224))  # ปรับขนาดเป็น 224x224

    # แปลงเป็น RGB หากรูปเป็น RGBA หรือรูปประเภทอื่น
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # ปรับสเกลให้เป็น [0, 1]
    return img_array

def show():
    st.title("🧚‍♀️ Princess Classifier - Neural Network")
    
    # กำหนดการแสดงผลปุ่มและการสุ่มรูปภาพ
    if sample_images:
        # สุ่มรูปเจ้าหญิง Disney ทุกครั้งที่กดปุ่ม
        selected_label, selected_image_path = random.choice(sample_images)

        try:
            img = Image.open(selected_image_path)
            
            # ตรวจสอบว่าไฟล์เป็นประเภทที่สามารถใช้ได้ (ไม่เป็น .png)
            if selected_image_path.endswith('.png'):
                st.error(f"🚨 ไฟล์ {selected_image_path} เป็น .png ไม่สามารถใช้ได้โปรดลองรูปอื่น.")
                return  # ไม่ให้แสดงปุ่มในกรณีนี้
            
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
        
        except Exception as e:
            st.error(f"🚨 ไม่สามารถประมวลผลรูปภาพนี้: {selected_image_path} เนื่องจาก: {e}")
    
    # ให้ปุ่มแรนด้อมแสดงอยู่ตลอดเวลา
    if "clicked" not in st.session_state or not st.session_state.clicked:
        st.session_state.clicked = True
        st.button("🔀 ทำนายรูปเจ้าหญิง Disney ใหม่", on_click=show)

# เรียกฟังก์ชันแสดงผล
show()
