import os
import numpy as np
import tensorflow as tf
import logging
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
DATA_PATH = os.path.join(BASE_DIR, "../features/")  # Ensure features are stored in a separate folder
MODEL_DIR = os.path.join(BASE_DIR, "../models/")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
FEATURES_FILE = os.path.join(DATA_PATH, "features.npy")
LABELS_FILE = os.path.join(DATA_PATH, "labels.npy")

# Check if feature and label files exist
if not os.path.exists(FEATURES_FILE) or not os.path.exists(LABELS_FILE):
    logging.error("Missing dataset files. Ensure feature extraction is complete.")
    exit(1)

# Load features and labels
logging.info("Loading feature and label data...")
X = np.load(FEATURES_FILE, allow_pickle=False)
y = np.load(LABELS_FILE, allow_pickle=False)

# Debugging Info
logging.info(f"Feature data shape: {X.shape}, dtype: {X.dtype}")
logging.info(f"Label data shape: {y.shape}, dtype: {y.dtype}")

# Normalize Features (Standardization: mean=0, std=1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Convert labels to categorical (One-Hot Encoding)
num_classes = len(set(y))
y = tf.keras.utils.to_categorical(y, num_classes)

logging.info("Data loaded and preprocessed successfully.")

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definition
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
logging.info("Starting model training...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32,
                    verbose=1)

# Generate timestamped model name to prevent overwriting
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_PATH = os.path.join(MODEL_DIR, f"speech_auth_model_{timestamp}.h5")

# Save Model
model.save(MODEL_PATH)
logging.info(f"Model saved successfully at {MODEL_PATH}")

# Plot Training Performance
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, f"training_plot_{timestamp}.png"))
plt.show()
logging.info("Training visualization saved.")
