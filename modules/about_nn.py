import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    """ Display the process of developing a Neural Network and the code used """
    st.title("📖 Neural Networks Development Guide")
    st.write("This page explains the process of developing a Neural Network, from data preparation to model development")

    st.image("datasources/1.gif", use_container_width=600)
    
    # Step 1: Data Preparation
    st.subheader("🔹 Data Preparation")
    st.write(""" 
    Before training the model, we need to prepare the data so that it's in the right format for the model
    - Use a **dataset** from sources such as Kaggle
    - **Remove duplicate** or corrupted data
    - **Resize images** to a standard size, such as 224x224 pixels
    - **Apply Data Augmentation** such as flipping images or adjusting brightness
    """)

    # Display sample images of Disney Princesses (e.g., Belle, Aurora)
    st.write("Here are sample images of Disney Princesses we will use for training:")


    st.code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preparation for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to be between 0-1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True  # Flip images horizontally
)

test_datagen = ImageDataGenerator(rescale=1./255)  # For testing data
    """, language="python")

    # Step 2: Model Creation
    st.subheader("🔹 Model Creation")
    st.write("""
    The Neural Network model consists of multiple layers, each responsible for different operations that allow the model to learn from the data
    - **Input Layer** accepts input data
    - **Hidden Layers** use activation functions like **ReLU** and **Dropout** to reduce Overfitting
    - **Output Layer** produces predictions, using **Softmax** for multiple classes
    - **Loss Function** such as **Categorical Crossentropy** is used to calculate the loss
    """)

    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Creating the Neural Network model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),  # Conv2D layer
    MaxPooling2D(pool_size=(2,2)),  # MaxPooling2D layer
    Flatten(),  # Flatten the data from 2D to 1D
    Dense(128, activation='relu'),  # Dense layer
    Dropout(0.5),  # Dropout layer to reduce Overfitting
    Dense(10, activation='softmax')  # Output layer for 10 classes
])
    """, language="python")

    # Step 3: Model Training
    st.subheader("🔹 Model Training")
    st.write("""
    This step trains the model with the prepared data. The model will learn from the data to make accurate predictions
    - Set the **optimizer** such as **Adam**
    - Use an appropriate **Loss Function** like **Categorical Crossentropy**
    - Specify the number of **epochs** to train the model
    """)

    st.code("""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model with training and validation data
model.fit(train_generator, validation_data=validation_generator, epochs=10)
    """, language="python")

    # Step 4: Model Testing
    st.subheader("🔹 Model Testing")
    st.write("""
    After training the model, we need to test it to evaluate its performance and accuracy.
    Using **validation data** and **testing data** helps us understand how well the model performs.
    """)

    # Step 5: Model Tuning
    st.subheader("🔹 Model Tuning")
    st.write("""
    Model tuning is essential to improve the model's performance. For example, using **Dropout** to reduce Overfitting or choosing the best **optimizer**.
    We can adjust hyperparameters such as **Learning Rate** or **Batch Size** to achieve the best model performance.
    """)

    # Display additional image after content
    st.image("datasources/2.gif", use_container_width=600)

if __name__ == "__main__":
    show()
