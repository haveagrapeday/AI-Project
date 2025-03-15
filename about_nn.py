import streamlit as st

def show():
    """ แสดงแนวทางการพัฒนา Neural Network และโค้ดที่ใช้ """
    st.title("📖 About Neural Networks")
    st.write("หน้านี้อธิบายแนวทางการพัฒนาโมเดล Neural Network ตั้งแต่การเตรียมข้อมูลไปจนถึงการพัฒนาโมเดล")

    # 🔹 1. การเตรียมข้อมูล
    st.subheader("🔹 Data Preparation")
    st.write("""
    - นำ **dataset จาก Kaggle** หรือแหล่งข้อมูลอื่น  
    - **ลบข้อมูลซ้ำ** และ **ปรับขนาดภาพเป็น 224x224**  
    - **เพิ่มข้อมูลเสริม (Data Augmentation)** เช่น **พลิกภาพ, ปรับแสง**  
    """)
    st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
    """, language="python")

    # 🔹 2. ทฤษฎี Neural Network
    st.subheader("🔹 Neural Network Theory")
    st.write("""
    - **Input Layer:** รับข้อมูลภาพที่ผ่านการปรับขนาด  
    - **Hidden Layers:** ใช้ **ReLU Activation** และ **Dropout** เพื่อลด Overfitting  
    - **Output Layer:** ใช้ **Softmax Activation** สำหรับคลาสที่มากกว่า 2  
    - ใช้ **Loss Function** เช่น **Categorical Crossentropy**
    """)
    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
    """, language="python")

    # 🔹 3. ขั้นตอนการพัฒนาโมเดล
    st.subheader("🔹 NN Model Development Steps")
    st.write("""
    1️⃣ **เตรียมข้อมูล** (โหลดข้อมูล, แปลงภาพ)  
    2️⃣ **กำหนดโครงสร้างโมเดล** (Layers, Activation Functions)  
    3️⃣ **ฝึกโมเดล** (Training)  
    4️⃣ **ทดสอบโมเดล** (Validation & Testing)  
    5️⃣ **ปรับแต่งโมเดล** เพื่อลด Overfitting  
    """)
    st.code("""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)
    """, language="python")

if __name__ == "__main__":
    show()
