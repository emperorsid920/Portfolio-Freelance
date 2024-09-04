import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.model_selection import train_test_split
import numpy as np
import os
from pathlib import Path
import math

# Set up paths
data_dir = Path("Dataset")
categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load and preprocess images with larger size (150x150)
def load_images(data_dir, categories):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        category_path = data_dir / category
        for img_path in category_path.glob("*.jpg"):
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))  # Larger image size
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

# Data Augmentation with reduced transformations
datagen = ImageDataGenerator(
    rotation_range=20,  # Reduced rotation range
    zoom_range=0.1,     # Reduced zoom range
    horizontal_flip=True
)

# Load the VGG16 model with ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the convolutional base

# Build the model with less regularization and smaller dropout
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Removed L2 Regularization and decreased neurons
    layers.Dropout(0.4),  # Reduced Dropout
    layers.Dense(64, activation='relu'),  # Another Dense layer without L2
    layers.Dropout(0.3),
    layers.Dense(len(categories), activation='softmax')
])

# Cosine decay learning rate scheduler
initial_learning_rate = 0.0001
def cosine_decay(epoch):
    return initial_learning_rate * (0.5 * (1 + math.cos(math.pi * epoch / 10)))

lr_scheduler = LearningRateScheduler(cosine_decay)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, learning_rate_reduction, lr_scheduler]
)

# Fine-tuning: Unfreeze the base model and train further with a lower learning rate
base_model.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
history_finetune = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, learning_rate_reduction, lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")


# Save the model in HDF5 format
model.save('/content/drive/MyDrive/Portfolio Freelance/Recycle/recycle_model.h5')

