import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, Concatenate, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched_img = ((img - min_val) / (max_val - min_val)) * 255
    return stretched_img.astype(np.uint8)

def load_images_and_labels(parent_folder):
    images = []
    labels = []
    cloud_distributions = []

    images_folder = os.path.join(parent_folder, 'images')
    labels_csv_path = os.path.join(parent_folder, 'labels.csv')
    cloud_distributions_csv_path = os.path.join(parent_folder, 'cloud_distributions.csv')

    # Load images
    for img_name in os.listdir(images_folder):
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_stretched = contrast_stretching(img)

        _, thresh = cv2.threshold(img_stretched, 128, 255, cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        img_resized = cv2.resize(dilation, (128, 128))
        img_normalized = img_resized / 255.0

        images.append(img_normalized)

    # Load labels and cloud distributions from CSV files
    labels_data = pd.read_csv(labels_csv_path)
    cloud_distributions_data = pd.read_csv(cloud_distributions_csv_path)

    # Process labels and cloud distributions
    for img_name in os.listdir(images_folder):
        label_row = labels_data[labels_data['Filename'] == img_name]
        cloud_distribution_row = cloud_distributions_data[cloud_distributions_data['Filename'] == img_name]

        if not label_row.empty and not cloud_distribution_row.empty:
            label = label_row['Label'].values[0]
            cloud_distribution = cloud_distribution_row['Cloud Distribution'].values[0]
            labels.append(label)
            cloud_distributions.append(cloud_distribution)
        else:
            print(f"No label or cloud distribution found for image: {img_name}")

    # Convert lists to NumPy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)
    cloud_distributions_array = np.array(cloud_distributions)

    # Reshape images array to include a new axis
    images_array = images_array[:, :, :, np.newaxis]

    return images_array, labels_array, cloud_distributions_array

parent_folder = r'C:\Users\Asus\Desktop\KAPOYA'

# Load data
X_train, y_train, cloud_distribution_train = load_images_and_labels(os.path.join(parent_folder, 'training'))
X_val, y_val, cloud_distribution_val = load_images_and_labels(os.path.join(parent_folder, 'validation'))

# Continue with your model training, etc.

# Model Creation
def create_model(input_shape, num_features):
    input_img = Input(shape=input_shape, name='image_input')
    input_features = Input(shape=(num_features,), name='cloud_distribution_input')
    X = Conv2D(64, (3, 3), activation='relu')(input_img)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Flatten()(X)

    merged = Concatenate()([X, input_features])

    X = Dense(256, activation='relu')(merged)
    X = Dropout(0.5)(X)
    X = Dense(128, activation='tanh')(X)
    X = Dropout(0.5)(X)

    output = Dense(1, activation="linear")(X)

    model = Model(inputs=[input_img, input_features], outputs=output)

    return model

#Additional Metrics
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.keras.losses.MeanSquaredError()(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

model = create_model(input_shape=(128, 128, 1), num_features=1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=[root_mean_squared_error, 'mean_absolute_error', mean_absolute_percentage_error],
)

# Custom Data Generator
def custom_generator(X_img, cloud_distribution, y, batch_size):
    image_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    while True:
        idx = np.arange(len(X_img))
        np.random.shuffle(idx)
        for i in range(0, len(X_img), batch_size):
            # Training
            batch_idx = idx[i:i+batch_size]
            X_img_batch = X_img[batch_idx]
            cloud_distribution_batch = cloud_distribution[batch_idx]
            y_batch = y[batch_idx]
            # Perform image augmentation here
            X_img_batch = image_gen.flow(X_img_batch, batch_size=batch_size, shuffle=False).next()
            yield [X_img_batch, cloud_distribution_batch], y_batch

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

batch_size = 64
train_gen = custom_generator(X_train, cloud_distribution_train, y_train, batch_size)
steps_per_epoch = np.ceil(len(X_train) / batch_size)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=([X_val, cloud_distribution_val], y_val),
    epochs=100,
    callbacks=[early_stopping],
)

# Define the file path where you want to save the model
model_save_path = 'solar_irradiance_model.h5'

# Save the model
model.save(model_save_path)

print("Model saved successfully at:", model_save_path)
