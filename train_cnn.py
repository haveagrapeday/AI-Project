import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data(data_dir, img_size, batch_size):
    """Load training and validation data."""
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
    """Build the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Configurations
    DATA_DIR = "datasources/princess"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    MODEL_PATH = "disney_princess_model.h5"

    # Load Data
    train_generator, val_generator = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_generator.class_indices)

    # Build Model
    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)

    # Train Model
    model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

    # Save Model
    model.save(MODEL_PATH)
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    main()
