import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
from PIL import Image

def clean_dataset(data_dir):
    """ แปลงภาพพาเลตเป็น RGBA และลบภาพที่เสียหาย """
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
                if img.mode == "P":
                    img = img.convert("RGBA")
                    img.save(file_path)
            except Exception as e:
                print(f"⚠️ Removing corrupted image: {file_path}")
                os.remove(file_path)

def load_data(data_dir, img_size, batch_size):
    """ โหลดข้อมูลและใช้ Augmentation """
    clean_dataset(data_dir)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
    val_generator = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')
    return train_generator, val_generator

def compute_class_weights(generator):
    """ คำนวณน้ำหนักของแต่ละคลาส ถ้าข้อมูลไม่สมดุล """
    counter = Counter(generator.classes)
    max_count = max(counter.values())
    class_weights = {cls: max_count / count for cls, count in counter.items()}
    return class_weights

def build_model(input_shape, num_classes):
    """ สร้างโมเดล ResNet50 พร้อม Fine-tuning """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-30]:  # Unfreeze เฉพาะ 30 ชั้นสุดท้าย
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    DATA_DIR = "datasources/princess"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 30
    MODEL_PATH = "disney_princess_resnet50_best.h5"
    
    print("📂 Loading data...")
    train_generator, val_generator = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    print(f"📊 Classes found: {train_generator.class_indices}")
    
    class_weights = compute_class_weights(train_generator)
    
    print("🛠️ Building model..")
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3), len(train_generator.class_indices))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    print("🚀 Training model...")
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, class_weight=class_weights, callbacks=callbacks)
    print("✅ Training complete!")
    print(f"💾 Model saved at {MODEL_PATH}")
    
if __name__ == "__main__":
    main()

