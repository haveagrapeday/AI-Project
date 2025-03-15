import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.title("🧚‍♀️ Princess Classifier - Neural Network")

MODEL_PATH = "disney_princess_resnet50.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# โหลด class labels
DATA_DIR = "datasources/princess"
class_labels = sorted(os.listdir(DATA_DIR))

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

uploaded_file = st.file_uploader("📤 อัปโหลดรูปเจ้าหญิง Disney", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 รูปที่อัปโหลด", use_column_width=True)
    
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predi
