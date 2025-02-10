# import os
# import librosa
# import numpy as np

# # Define paths
# DATA_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"
# SAVE_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\data"

# # Function to extract MFCC features
# def extract_mfcc(audio, sr, n_mfcc=13):
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     return np.mean(mfcc, axis=1)  # Taking mean over time

# # Initialize lists
# features = []
# labels = []

# # Check if the data folder exists
# if not os.path.exists(DATA_PATH):
#     print(f"ERROR: Data folder {DATA_PATH} does not exist!")
#     exit(1)

# # Process audio files
# for filename in os.listdir(DATA_PATH):
#     if filename.endswith(".wav"):  # Ensure only processing audio files
#         file_path = os.path.join(DATA_PATH, filename)
#         print(f"Processing: {filename}")

#         try:
#             sample_audio, sample_rate = librosa.load(file_path, sr=22050)
#             feature = extract_mfcc(sample_audio, sample_rate)
#             features.append(feature)

#             # Extract label from filename (assuming label is first character)
#             label = int(filename.split("_")[0])  
#             labels.append(label)
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# # Convert to numpy arrays
# features = np.array(features)
# labels = np.array(labels)

# # Check if features were extracted
# if len(features) == 0 or len(labels) == 0:
#     print("ERROR: No features extracted. Check if .wav files exist in the folder.")
#     exit(1)

# # Save extracted features
# features_path = os.path.join(SAVE_PATH, "features.npy")
# labels_path = os.path.join(SAVE_PATH, "labels.npy")

# print(f"Saving features to: {features_path}")
# print(f"Saving labels to: {labels_path}")

# np.save(features_path, features)
# np.save(labels_path, labels)

# print("Features and labels saved successfully!")

import os
import librosa
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
DATA_PATH = os.path.join(BASE_DIR, "../data/")  # Folder containing .wav files
FEATURES_PATH = os.path.join(BASE_DIR, "../features/")  # Folder to save extracted features

# Ensure output directory exists
os.makedirs(FEATURES_PATH, exist_ok=True)

def extract_mfcc(audio, sr, n_mfcc=13):
    """Extract MFCC features and take mean over time."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Reduce time dimension

def process_audio_files():
    """Process all audio files in DATA_PATH and extract MFCC features."""
    features = []
    labels = []

    # Check if data folder exists
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data folder not found: {DATA_PATH}")
        return None, None

    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".wav")]
    
    if not file_list:
        logging.warning("No .wav files found in the data directory!")
        return None, None

    for filename in file_list:
        file_path = os.path.join(DATA_PATH, filename)
        logging.info(f"Processing: {filename}")

        try:
            sample_audio, sample_rate = librosa.load(file_path, sr=22050)
            feature = extract_mfcc(sample_audio, sample_rate)
            features.append(feature)

            # Extract label (assuming label is first part of filename before "_")
            label = filename.split("_")[0]
            labels.append(label)
        
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    features, labels = process_audio_files()

    if features is None or labels is None:
        logging.error("No features extracted. Please check your data files.")
        exit(1)

    # Save extracted features
    np.save(os.path.join(FEATURES_PATH, "features.npy"), features)
    np.save(os.path.join(FEATURES_PATH, "labels.npy"), labels)

    logging.info("Feature extraction complete. Files saved successfully!")
