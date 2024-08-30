import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the path to the dataset directory (relative path)
dataset_dir = "Dataset"
categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# Function to load and preprocess images
def load_images(dataset_dir, categories, img_size=(224, 224)):
    data = []
    labels = []

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        label = categories.index(category)  # Assign a numeric label to each category
        print(f"Loading images from category: {category}")

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            try:
                img = Image.open(img_path)
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0

                data.append(img_array)
                labels.append(label)

            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    return np.array(data), np.array(labels)


# Load and preprocess the images
train_images, train_labels = load_images(dataset_dir, categories)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # 6 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
