# import numpy as np
# import os
# import tensorflow as tf 

# # Define the correct path to the data folder
# DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"

# # Construct full file paths
# features_file = os.path.join(DATA_PATH, "features.npy")

# labels_file = os.path.join(DATA_PATH, "labels.npy")

# # Print debug information
# print(f"Checking for: {features_file}")
# print(f"Checking for: {labels_file}")

# # Verify that files exist before loading
# if not os.path.exists(features_file):
#     raise FileNotFoundError(f"ERROR: File not found: {features_file}")

# if not os.path.exists(labels_file):
#     raise FileNotFoundError(f"ERROR: File not found: {labels_file}")

# # Load feature and label data
# try:
#     print("Loading feature and label data...")
#     X = np.load(features_file, allow_pickle=False)  # Disable pickle for security
#     y = np.load(labels_file, allow_pickle=False)

#     # Print shape info
#     print(f"Feature data shape: {X.shape}, dtype: {X.dtype}")
#     print(f"Label data shape: {y.shape}, dtype: {y.dtype}")

#     # Normalize Features
#     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardization

#     # Convert labels to categorical (One-Hot Encoding)
#     num_classes = len(set(y))
#     y = tf.keras.utils.to_categorical(y, num_classes)

#     print("Loaded data successfully!")

# except Exception as e:
#     print(f"ERROR: Failed to load data: {e}")
#     exit(1)

# # Import ML/DL libraries
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam as AdamOptimizer 
# from sklearn.model_selection import train_test_split

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Definition
# model = Sequential([
#     Dense(256, activation='relu', input_shape=(X.shape[1],)),
#     BatchNormalization(),
#     Dropout(0.4),

#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),

#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),

#     Dense(num_classes, activation='softmax')  # Output layer for classification
# ])

# # Compile the model
# model.compile(optimizer=AdamOptimizer(learning_rate=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# print("Training the model...")
# history = model.fit(X_train, y_train,
#                     validation_data=(X_val, y_val),
#                     epochs=30,
#                     batch_size=32)

# # Save the trained model
# MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_authentication_model.h5"
# print(f"Saving model to {MODEL_PATH}...")
# model.save(MODEL_PATH)
# print("Model training complete and saved successfully!")

# train_model.py
import numpy as np
import os
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define paths
DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_authentication_model.h5"

# File paths
features_file = os.path.join(DATA_PATH, "features.npy")
labels_file = os.path.join(DATA_PATH, "labels.npy")

# Check if files exist
if not os.path.exists(features_file) or not os.path.exists(labels_file):
    raise FileNotFoundError("Missing dataset files. Ensure feature extraction has been done.")

# Load data
print("Loading feature and label data...")
X = np.load(features_file, allow_pickle=False)
y = np.load(labels_file, allow_pickle=False)

# Debugging Info
print(f"Feature data shape: {X.shape}, dtype: {X.dtype}")
print(f"Label data shape: {y.shape}, dtype: {y.dtype}")

# Normalize Features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Convert labels to categorical (One-Hot Encoding)
num_classes = len(set(y))
y = tf.keras.utils.to_categorical(y, num_classes)

print("Loaded data successfully!")

# Split data into training and validation sets
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

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32)

# Save the trained model
print(f"Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("Model training complete and saved successfully!")
