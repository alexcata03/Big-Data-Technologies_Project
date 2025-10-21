import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# Dataset paths
data_dir = "dataset"
val_dir = os.path.join(data_dir, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load validation dataset
val_ds_raw = image_dataset_from_directory(
    val_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)
class_names = val_ds_raw.class_names

# Count and print images per class
label_list = []
for _, labels in val_ds_raw:
    label_list.extend(labels.numpy())
counts = Counter(label_list)
print("\nClass distribution:")
for i, count in counts.items():
    print(f"{class_names[i]}: {count} images")

# Store all samples for randomized selection
all_images, all_labels = [], []
for images, labels in val_ds_raw:
    all_images.extend(images)
    all_labels.extend(labels)
all_images = np.array([img.numpy() for img in all_images])
all_labels = np.array(all_labels)

# Show random samples
indices = np.random.choice(len(all_images), 9, replace=False)
plt.figure(figsize=(10, 10))
for i, idx in enumerate(indices):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(all_images[idx].astype("uint8"))
    plt.title(class_names[all_labels[idx]])
    plt.axis("off")
plt.tight_layout()
plt.show()

# Normalize dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
val_ds = val_ds.batch(BATCH_SIZE).map(lambda x, y: (normalization_layer(x), y))

# Load model
model = load_model("fruit_classifier_mobilenetv2.h5")

# Predict and collect labels
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Show confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, colorbar=True)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.tight_layout()
plt.show()

# Shuffle dataset before creating prediction batches
shuffled_indices = np.random.permutation(len(all_images))
shuffled_images = all_images[shuffled_indices]
shuffled_labels = all_labels[shuffled_indices]

# Create a shuffled, normalized dataset
val_vis = tf.data.Dataset.from_tensor_slices((shuffled_images, shuffled_labels))
val_vis = val_vis.batch(BATCH_SIZE).map(lambda x, y: (normalization_layer(x), y))

# Display 4 batches (can be adjusted)
num_batches_to_display = 4

# Number of images to show per batch(can be adjusted)
images_per_batch = 16
grid_size = int(np.ceil(np.sqrt(images_per_batch)))

for batch_idx, (images, labels) in enumerate(val_vis.take(num_batches_to_display)):
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i in range(min(images_per_batch, len(images))):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        img = (images[i].numpy() * 255).astype("uint8")
        plt.imshow(img)
        actual = class_names[labels[i].numpy()]
        predicted = class_names[pred_classes[i]]
        color = "green" if predicted == actual else "red"
        plt.title(f"T: {actual}\nP: {predicted}", color=color, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
