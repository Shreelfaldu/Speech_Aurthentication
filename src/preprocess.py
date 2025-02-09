import os
import librosa
import numpy as np
import pandas as pd

# Use an absolute path to avoid issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of preprocess.py
DATA_DIR = os.path.join(BASE_DIR, "../data/")  # Adjust path to data folder

# Ensure the directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
# DATA_DIR = "../data/"  # Path to dataset

def load_audio(file_path, sr=22050):
    """Load an audio file and return the waveform and sample rate."""
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def normalize_audio(audio):
    """Normalize audio to have zero mean and unit variance."""
    return (audio - np.mean(audio)) / np.std(audio)

def process_dataset():
    """Load and normalize all audio files in the dataset."""
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    data = []

    for file_name in file_list:
        file_path = os.path.join(DATA_DIR, file_name)
        audio, sr = load_audio(file_path)
        audio = normalize_audio(audio)
        data.append((file_name, audio, sr))

    return pd.DataFrame(data, columns=["File", "Audio", "SampleRate"])

if __name__ == "__main__":
    df = process_dataset()
    print(df.head())
