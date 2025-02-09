# import os
# import numpy as np
# import librosa

# # Set paths
# DATA_DIR = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
# FEATURES_FILE = os.path.join(DATA_DIR, "features.npy")
# LABELS_FILE = os.path.join(DATA_DIR, "labels.npy")

# def extract_mfcc(audio, sr, n_mfcc=13):
#     """Extract MFCC features from an audio file."""
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     return np.mean(mfcc, axis=1)  # Mean across time

# # Initialize lists
# features = []
# labels = []

# # Process all WAV files in the data folder
# for filename in os.listdir(DATA_DIR):
#     if filename.endswith(".wav"):
#         file_path = os.path.join(DATA_DIR, filename)
#         print(f"Processing: {filename}")
#         try:
#             # Load audio
#             audio, sr = librosa.load(file_path, sr=22050)
#             mfcc_features = extract_mfcc(audio, sr)
            
#             # Append features and labels
#             features.append(mfcc_features)
#             labels.append(filename)  # Using filename as a label (change as needed)
        
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# # Convert lists to numpy arrays
# features = np.array(features)
# labels = np.array(labels)

# # Save to .npy files
# np.save(FEATURES_FILE, features)
# np.save(LABELS_FILE, labels)

# print(f"Features saved to {FEATURES_FILE}")
# print(f"Labels saved to {LABELS_FILE}")

import os
import librosa
import numpy as np

# Define paths
DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
SAVE_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"

# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Taking mean over time

# Initialize lists
features = []
labels = []

# Check if the data folder exists
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Data folder {DATA_PATH} does not exist!")
    exit(1)

# Process audio files
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".wav"):  # Ensure only processing audio files
        file_path = os.path.join(DATA_PATH, filename)
        print(f"Processing: {filename}")

        try:
            sample_audio, sample_rate = librosa.load(file_path, sr=22050)
            feature = extract_mfcc(sample_audio, sample_rate)
            features.append(feature)

            # Extract label from filename (assuming label is first character)
            label = int(filename.split("_")[0])  
            labels.append(label)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Check if features were extracted
if len(features) == 0 or len(labels) == 0:
    print("ERROR: No features extracted. Check if .wav files exist in the folder.")
    exit(1)

# Save extracted features
features_path = os.path.join(SAVE_PATH, "features.npy")
labels_path = os.path.join(SAVE_PATH, "labels.npy")

print(f"Saving features to: {features_path}")
print(f"Saving labels to: {labels_path}")

np.save(features_path, features)
np.save(labels_path, labels)

print("Features and labels saved successfully!")
