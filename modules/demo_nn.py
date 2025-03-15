import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

MODEL_PATH = "disney_princess_resnet50.h5"

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"🚨 โมเดลไม่สามารถโหลดได้: {e}")
        return None

def save_model(model):
    model.save(MODEL_PATH)
    st.success(f"✅ โมเดลถูกบันทึกแล้วที่ {MODEL_PATH}")

# Load model if available
model = load_model()

DATA_DIR = "datasources/princess"
class_labels = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else []

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Data Generator Setup for Training
def setup_data_generator():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator

# Train the model function
def train_model(train_generator):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_labels), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_generator, epochs=10, steps_per_epoch=100)  # Adjust based on your data size
    save_model(model)
    
    return history

# Streamlit app interface
def main():
    st.title("🧚‍♀️ Princess Classifier - Neural Network")
    
    action = st.radio("What would you like to do?", ["Predict", "Train Model"])

    if action == "Predict":
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

   