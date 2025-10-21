import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Set the paths to the dataset directories
data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")  # Path to training images
val_dir = os.path.join(data_dir, "test")     # Path to validation/test images

# Set the image input size and batch size
IMG_SIZE = (224, 224)  # Resize all images to 224x224 (default for MobileNetV2)
BATCH_SIZE = 32        # Number of images to load per batch
EPOCHS = 5             # Number of times the model will iterate over the full dataset

# Load the training dataset from folders and automatically assign labels from folder names
train_ds = image_dataset_from_directory(
    train_dir,
    shuffle=True,             # Randomize order for training
    batch_size=BATCH_SIZE,    # Load images in batches of 32
    image_size=IMG_SIZE,      # Resize all images to 224x224
    seed=123                  # Ensure reproducibility
)

# Load the validation/test dataset without shuffling
val_ds = image_dataset_from_directory(
    val_dir,
    shuffle=False,            # Do not shuffle so we can evaluate accurately
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Extract the class names from the folder structure
class_names = train_ds.class_names

# Normalize image pixel values to the range [0, 1] using Rescaling layer
normalization_layer = layers.Rescaling(1./255)

# Apply normalization to both training and validation datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Add real-time data augmentation to make the model more robust
# Random horizontal flip, slight rotation, and zoom
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
    layers.RandomRotation(0.1),        # Rotate images by up to 10%
    layers.RandomZoom(0.1),            # Zoom in by up to 10%
])

# Apply data augmentation only to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Use TensorFlow's AUTOTUNE to automatically optimize performance
AUTOTUNE = tf.data.AUTOTUNE

# Improve performance by caching, shuffling, and prefetching
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load the MobileNetV2 model without the top classification layers
# Use weights pre-trained on ImageNet, which helps with transfer learning
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),  # Input size must include color channels
    include_top=False,            # Exclude the final dense layer
    weights='imagenet'            # Use weights learned from ImageNet
)

# Freeze the base model so we only train the top layers
base_model.trainable = False

# Add custom classification layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),           # Flatten feature maps into a vector
    layers.Dense(128, activation='relu'),      # Fully connected layer with ReLU activation
    layers.Dropout(0.3),                       # Dropout layer to reduce overfitting
    layers.Dense(len(class_names), activation='softmax')  # Final output layer for multi-class
])

# Compile the model with:
# - Adam optimizer (adaptive learning rate)
# - Sparse categorical crossentropy (suitable for integer labels)
# - Accuracy metric to monitor performance
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using the prepared datasets
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the final trained model to an H5 file
model.save("fruit_classifier_mobilenetv2.h5")
