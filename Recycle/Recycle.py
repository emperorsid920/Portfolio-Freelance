import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import os
from pathlib import Path

# Set up paths
data_dir = Path("/Users/sidkumar/Documents/Portfolio Freelance/Recycle/Dataset")
categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load and preprocess images with reduced size
def load_images(data_dir, categories):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        category_path = data_dir / category
        for img_path in category_path.glob("*.jpg"):
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(100, 100))  # Reduced image size
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

X, y = load_images(data_dir, categories)
X = X / 255.0  # Normalize pixel values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simplified data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Smaller rotation range
    zoom_range=0.1,     # Smaller zoom range
    horizontal_flip=True  # Only flip images horizontally
)

# Load the VGG16 model with ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))  # Update input shape
base_model.trainable = False  # Freeze the convolutional base

# Build the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Reduced the number of neurons
    layers.Dropout(0.5),  # Add Dropout
    layers.BatchNormalization(),  # Add Batch Normalization
    layers.Dense(len(categories), activation='softmax')  # Adjust output layer for your number of classes
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)

# Train the model with data augmentation and smaller batch size
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Smaller batch size
    epochs=10,  # Start with fewer epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, learning_rate_reduction]
)

# Fine-tuning: Unfreeze the base model and train further with a lower learning rate
base_model.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
history_finetune = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Smaller batch size
    epochs=10,  # Limit fine-tuning epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, learning_rate_reduction]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Optional: Predict and generate classification report if needed
y_pred = np.argmax(model.predict(X_test), axis=-1)
