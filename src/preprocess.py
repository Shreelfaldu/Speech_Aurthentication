# import os
# import librosa
# import numpy as np
# import pandas as pd

# # Use an absolute path to avoid issues
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of preprocess.py
# DATA_DIR = os.path.join(BASE_DIR, "../data/")  # Adjust path to data folder

# # Ensure the directory exists
# if not os.path.exists(DATA_DIR):
#     raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
# # DATA_DIR = "../data/"  # Path to dataset

# def load_audio(file_path, sr=22050):
#     """Load an audio file and return the waveform and sample rate."""
#     audio, sample_rate = librosa.load(file_path, sr=sr)
#     return audio, sample_rate

# def normalize_audio(audio):
#     """Normalize audio to have zero mean and unit variance."""
#     return (audio - np.mean(audio)) / np.std(audio)

# def process_dataset():
#     """Load and normalize all audio files in the dataset."""
#     file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
#     data = []

#     for file_name in file_list:
#         file_path = os.path.join(DATA_DIR, file_name)
#         audio, sr = load_audio(file_path)
#         audio = normalize_audio(audio)
#         data.append((file_name, audio, sr))

#     return pd.DataFrame(data, columns=["File", "Audio", "SampleRate"])

# if __name__ == "__main__":
#     df = process_dataset()
#     print(df.head())

import os
import librosa
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Use an absolute path to avoid issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of preprocess.py
DATA_DIR = os.path.join(BASE_DIR, "../data/")  # Adjust path to data folder
FEATURES_DIR = os.path.join(BASE_DIR, "../features/")  # Path to save extracted features

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

def load_audio(file_path, sr=22050):
    """Load an audio file and return the waveform and sample rate."""
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def normalize_audio(audio):
    """Normalize audio to have zero mean and unit variance."""
    return (audio - np.mean(audio)) / np.std(audio)

def extract_features(audio, sr):
    """Extract Mel Spectrogram, MFCCs, and Chromagram from an audio signal."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    return mel_spec_db, mfccs, chroma

def process_dataset():
    """Load, normalize, extract features, and save them for all audio files."""
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    data = []

    for file_name in file_list:
        file_path = os.path.join(DATA_DIR, file_name)
        logging.info(f"Processing {file_name}...")

        # Load and normalize audio
        audio, sr = load_audio(file_path)
        audio = normalize_audio(audio)

        # Extract features
        mel_spec, mfccs, chroma = extract_features(audio, sr)

        # Save features as numpy files
        np.save(os.path.join(FEATURES_DIR, f"{file_name}_mel.npy"), mel_spec)
        np.save(os.path.join(FEATURES_DIR, f"{file_name}_mfcc.npy"), mfccs)
        np.save(os.path.join(FEATURES_DIR, f"{file_name}_chroma.npy"), chroma)

        data.append((file_name, sr, mel_spec.shape, mfccs.shape, chroma.shape))

    return pd.DataFrame(data, columns=["File", "SampleRate", "Mel_Spec_Shape", "MFCC_Shape", "Chroma_Shape"])

if __name__ == "__main__":
    df = process_dataset()
    logging.info("Feature extraction complete. Processed files:")
    logging.info(df)
