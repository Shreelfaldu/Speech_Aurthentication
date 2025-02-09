import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "../models/speech_auth_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def authenticate(audio_path):
    """Authenticate user based on voice input."""
    audio, sr = librosa.load(audio_path, sr=22050)
    features = np.expand_dims(np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1), axis=0)
    
    prediction = np.argmax(model.predict(features), axis=1)
    return prediction

if __name__ == "__main__":
    user_audio = "../data/test_user.wav"
    result = authenticate(user_audio)
    print("Authenticated User ID:", result)
