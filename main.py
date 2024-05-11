import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Path to the training and validation folders containing image files
training_folder = r"KAPOYA\training"
validation_folder = r"KAPOYA\validation"

# Load labels from csv file
training_label_file = pd.read_csv(os.path.join(training_folder, "output.csv"))
validation_label_file = pd.read_csv(os.path.join(validation_folder, "output.csv"))

# Initialize empty lists to store training and validation images and labels
X_train = []
y_train = []

X_val = []
y_val = []

# Function for loading and preprocessing images with basic progress bar
def load_and_preprocess_images(folder, label_file, X, y):
    total_images = len(label_file)
    for i, row in enumerate(label_file.itertuples(), 1):
        image_path = os.path.join(folder, "images", row.Filename)
        image = cv2.imread(image_path)

        if image is None:
            print(image_path)
            continue

        image = cv2.resize(image, (224, 224))  # Resize image to VGG16 input size
        image = image / 255.0  # Normalize pixel values
        X.append(image)
        y.append(row.Label)

        # Print basic progress bar
        progress = i / total_images
        progress_bar = "=" * int(progress * 50) + "-" * (50 - int(progress * 50))
        print(f"[{progress_bar}] {i}/{total_images}", end="\r")

# Load and preprocess images with basic progress bar
load_and_preprocess_images(training_folder, training_label_file, X_train, y_train)
load_and_preprocess_images(validation_folder, validation_label_file, X_val, y_val)

# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers on top of the base model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='linear')

# Create a Sequential model with VGG16 base and custom classification layers
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with progress bar and plot the training and validation loss
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

# Plot training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
loss, mae = model.evaluate(X_val, y_val)

print("Mean Squared Error:", loss)
print("Mean Absolute Error:", mae)
