import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ตั้งค่าพาธโมเดลและข้อมูล
DATA_DIR = "C:/Users/uoobu/Desktop/Final/Fianl-project-AI/AI-project/datasources/princess"
MODEL_PATH = "C:/Users/uoobu/Desktop/Final/Fianl-project-AI/AI-project/disney_princess_model.h5"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Anna", "Ariel", "Aurora", "Belle", "Cinderella", "Elsa", "Jasmine", "Merida", "Rapunzel", "Snow White", "Tiana"]

def load_dataset():
    """ โหลดและแสดงตัวอย่างข้อมูล """
    st.subheader("🔹 Sample Data from Dataset")
    
    if not os.path.exists(DATA_DIR):
        st.error(f"Error: Directory '{DATA_DIR}' not found.")
        return

    # Data Generator
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=5,
        class_mode='categorical'
    )

    # แสดงตัวอย่างภาพ
    images, labels = next(generator)
    fig, axes = plt.subplots(1, 5, figsize=(10, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
        ax.set_title(CLASS_NAMES[np.argmax(labels[i])])
    st.pyplot(fig)

def predict_image(image_path, model):
    """ ทำนายภาพ """
    image = load_img(image_path, target_size=IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_label = CLASS_NAMES[np.argmax(prediction)]
    
    return predicted_label, prediction

def show():
    st.title("Neural Network Demo")
    st.write("🔹 This page displays the dataset loading process and prediction results.")

    # โหลดและแสดงข้อมูลตัวอย่าง
    load_dataset()

    # โหลดโมเดล
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return

    # อัปโหลดและทำนายภาพ
    st.subheader("🔹 Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # บันทึกไฟล์ชั่วคราว
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ทำนายภาพ
        predicted_label, prediction = predict_image(temp_path, model)

        # แสดงผลลัพธ์
        st.write(f"### 🎯 Prediction: **{predicted_label}**")
        st.bar_chart(prediction[0])  # แสดงผลลัพธ์แต่ละคลาสเป็นกราฟ

if __name__ == "__main__":
    show()