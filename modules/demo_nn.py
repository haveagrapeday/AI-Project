import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

MODEL_PATH = "disney_princess_resnet50.h5"
DATA_DIR = "datasources/princess"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"🚨 โมเดลไม่สามารถโหลดได้: {e}")
    return None

# Load model if available
model = load_model()
class_labels = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def main():
    st.title("🧚‍♀️ Princess Classifier - Neural Network")
    
    if model is None:
        st.error("🚨 โมเดลไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดล")
        return
    
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    selected_image = st.selectbox("📂 เลือกรูปภาพจากโฟลเดอร์", image_files)
    
    if selected_image:
        img_path = os.path.join(DATA_DIR, selected_image)
        img = Image.open(img_path)
        st.image(img, caption="📸 รูปที่เลือก", use_column_width=True)
        
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        
        if class_labels:
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.write(f"✨ เจ้าหญิงที่ทำนายได้: **{predicted_class}**")
            st.write(f"🎯 ความมั่นใจ: **{confidence:.2f}%**")
        else:
            st.error("🚨 ไม่พบข้อมูล class labels! กรุณาตรวจสอบโฟลเดอร์ `datasources/princess`")

if __name__ == "__main__":
    main()
