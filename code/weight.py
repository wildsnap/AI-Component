# pip install mlflow tensorflow numpy matplotlib seaborn scikit-learn pillow opencv-python

import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import ssl

from collections import Counter

from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

from sklearn.metrics import classification_report, confusion_matrix


ssl._create_default_https_context = ssl._create_unverified_context


# =========================
# CONFIGURATION
# =========================
BASE_DIR = "../wildsnap-dataset"
BATCH_SIZE = 32
EPOCHS = 10

# =========================
# Backbone Models
# =========================
backbones = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB4": EfficientNetB4,
}

recommended_sizes = {
    "MobileNetV2": (224, 224),
    "EfficientNetB0": (224, 224),
    "EfficientNetB4": (380, 380),
}

results = {}

# =========================
# Basic Input Data Exploration
# =========================
train_dir = os.path.join(BASE_DIR, 'train')
class_counts = {class_name: len(os.listdir(os.path.join(train_dir, class_name)))
                for class_name in os.listdir(train_dir)}

print("Class Distribution in Train Set:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")

# Show image shape and type
sample_path = os.path.join(train_dir, list(class_counts.keys())[0], os.listdir(os.path.join(train_dir, list(class_counts.keys())[0]))[0])
sample_img = cv2.imread(sample_path)
print(f"Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}")


# =========================
# Logging to MLflow
# =========================
def logMlFlow(model, name, img_size, epochs, test_acc, test_loss):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        mlflow.log_param("image_size", img_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, artifact_path="model")

# =========================
# Evaluation Report
# =========================
def classReport(model, name, test_gen):
    pred_probs = model.predict(test_gen)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))

# =========================
# Main Training Loop
# =========================
for name, backbone in backbones.items():
    print(f"\nTraining {name}...\n")

    IMG_SIZE = recommended_sizes[name]

    # Use EfficientNet-specific preprocessing
    if "EfficientNet" in name:
        train_datagen = ImageDataGenerator(preprocessing_function=effnet_preprocess)
        val_datagen = ImageDataGenerator(preprocessing_function=effnet_preprocess)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)

    # Data Loaders
    train_gen = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = val_datagen.flow_from_directory(
        os.path.join(BASE_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    base_model = backbone(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[es])

    # Evaluation
    test_loss, test_acc = model.evaluate(test_gen)
    results[name] = round(test_acc * 100, 3)

    # MLflow logging
    logMlFlow(model, name, IMG_SIZE, EPOCHS, test_acc, test_loss)
    classReport(model, name, test_gen)

    model.save(f"{name}.h5")
    print(f"Saved model: {name}.h5")

# =========================
# Final Comparison
# =========================
best_model = max(results, key=results.get)

print("\nFinal Accuracy Comparison:")
for model_name, acc in results.items():
    print(f"{model_name}: {acc}%")

print(f"\nBest model: {best_model}")
mlflow.log_param("best_model", best_model)

# Plot
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='lightgreen')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("model_accuracy_comparison.png")
plt.close()
