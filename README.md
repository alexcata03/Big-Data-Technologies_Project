Fruit Image Classification

This project uses transfer learning with MobileNetV2 (TensorFlow/Keras) to classify fruits as fresh or rotten. The model is trained on the Fruits – Fresh and Rotten for Classification dataset
, applying real-time data augmentation (flip, rotation, zoom) and optimized input pipelines using caching and prefetching.

The pipeline:

Loads and preprocesses images from foldered datasets (train/, test/)

Uses MobileNetV2 as a frozen feature extractor with custom dense layers on top

Trains for multiple epochs and saves the model as fruit_classifier_mobilenetv2.h5

Evaluates results with a classification report, confusion matrix, and visual prediction grids

Tech stack: Python, TensorFlow, Keras, NumPy, Matplotlib, scikit-learn
