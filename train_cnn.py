import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image

def convert_image_to_rgba(image_path):
    """แปลงภาพพาเลตให้เป็น RGBA เพื่อป้องกันข้อผิดพลาด"""
    try:
        img = Image.open(image_path)
        if img.mode == "P":  # ถ้าเป็นพาเลต
            img = img.convert("RGBA")  # แปลงเป็น RGBA
            img.save(image_path)  # บันทึกทับไฟล์เดิม
    except Exception as e:
        print(f"❌ Error converting {image_path}: {e}")

def clean_dataset(data_dir):
    """แปลงภาพพาเลตเป็น RGBA และลบภาพที่เสียหาย"""
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # ตรวจสอบว่าเปิดไฟล์ได้
                convert_image_to_rgba(file_path)  # แปลงเป็น RGBA ถ้าจำเป็น
            except Exception as e:
                print(f"⚠️ Warning: Removing corrupted image {file_path}")
                os.remove(file_path)  # ลบไฟล์ที่ไม่สามารถเปิดได้

def load_data(data_dir, img_size, batch_size):
    """โหลดข้อมูลภาพ"""
    print("🧹 กำลังตรวจสอบและแก้ไขภาพ...")
    clean_dataset(data_dir)  # ทำความสะอาดข้อมูลก่อนใช้งาน

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator

def build_model(input_shape, num_classes):
    """สร้างโมเดล ResNet50"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze โมเดลพื้นฐาน
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # ตั้งค่าพารามิเตอร์
    DATA_DIR = "datasources/princess"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 5
    MODEL_PATH = "disney_princess_resnet50.h5"

    print("📂 กำลังโหลดข้อมูล...")
    train_generator, val_generator = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"📊 พบ {len(train_generator.class_indices)} Classes: {train_generator.class_indices}")

    print("🛠️ กำลังสร้างโมเดล...")
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3), len(train_generator.class_indices))
    print("✅ โมเดลสร้างเสร็จ!")

    print("🚀 เริ่ม Training...")
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)
    print("✅ Training เสร็จ!")

    print("💾 บันทึกโมเดล...")
    model.save(MODEL_PATH)
    print(f"✅ โมเดลถูกบันทึกที่ {MODEL_PATH}")

if __name__ == "__main__":
    main()
