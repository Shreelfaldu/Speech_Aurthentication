# import tensorflow as tf
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score

# # Define paths
# DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
# MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_authentication_model.h5"  # Corrected path

# # File paths
# features_test_file = os.path.join(DATA_PATH, "features_test.npy")
# labels_test_file = os.path.join(DATA_PATH, "labels_test.npy")

# # Check if files exist
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# if not os.path.exists(features_test_file) or not os.path.exists(labels_test_file):
#     raise FileNotFoundError(f"Missing test dataset files. Run feature extraction for test data.")

# # Load model
# print(f"Loading model from {MODEL_PATH}...")
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load test data
# X_test = np.load(features_test_file)
# y_test = np.load(labels_test_file)

# print("Test data loaded successfully!")

# # Predict
# y_pred = np.argmax(model.predict(X_test), axis=1)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# #type 2
# import tensorflow as tf
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score

# # Define paths
# DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
# MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_authentication_model.h5"

# # File paths
# features_test_file = os.path.join(DATA_PATH, "features_test.npy")
# labels_test_file = os.path.join(DATA_PATH, "labels_test.npy")

# print(f"Expected test features file: {features_test_file}")
# print(f"Expected test labels file: {labels_test_file}")

# # Check if files exist
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# if not os.path.exists(features_test_file) or not os.path.exists(labels_test_file):
#     raise FileNotFoundError(f"Missing test dataset files. Run feature extraction for test data.")

# # Load model
# print(f"Loading model from {MODEL_PATH}...")
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load test data
# X_test = np.load(features_test_file)
# y_test = np.load(labels_test_file)

# # Ensure correct types
# X_test = X_test.astype(np.float32)

# print("Test data loaded successfully!")

# # Predict
# y_pred = np.argmax(model.predict(X_test), axis=1)

# # Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")


# test_model.py
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score

# Define paths
DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_authentication_model.h5"

# File paths
features_test_file = os.path.join(DATA_PATH, "features.npy")
labels_test_file = os.path.join(DATA_PATH, "labels.npy")


print(f"Expected test features file: {features_test_file}")
print(f"Expected test labels file: {labels_test_file}")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(features_test_file) or not os.path.exists(labels_test_file):
    raise FileNotFoundError("Missing test dataset files. Run feature extraction for test data.")

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
X_test = np.load(features_test_file)
y_test = np.load(labels_test_file)

# Debugging Info
print(f"Test feature shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"Test label shape: {y_test.shape}, dtype: {y_test.dtype}")
print(f"First 5 labels: {y_test[:5]}")

# Ensure correct types
X_test = X_test.astype(np.float32)

print("Test data loaded successfully!")

# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)

# Fix for AxisError (if y_test is already 1D, no need for argmax)
if len(y_test.shape) > 1:
    y_test = np.argmax(y_test, axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
