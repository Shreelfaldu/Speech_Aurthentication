import librosa
import numpy as np
import tensorflow as tf
import sys

# Define paths
MODEL_PATH = r"D:\work\Study\SEM_6\Project\speech_authentication\models\speech_auth_model.h5"

# Load the trained model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCC features (Ensure the same number of coefficients as used in training)
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Take the mean across time frames to get a fixed-size feature vector
        features = np.mean(mfcc_features, axis=1)
        
        # Ensure the correct shape (1, n_mfcc) for model prediction
        features = np.expand_dims(features, axis=0)
        
        return features
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        sys.exit(1)

def authenticate(audio_path):
    """Authenticate user based on voice input."""
    features = extract_features(audio_path)

    # Debugging Info
    print(f"Extracted features shape: {features.shape}")

    # Predict
    prediction = np.argmax(model.predict(features), axis=1)
    
    return prediction[0]  # Return the predicted user ID

if __name__ == "__main__":
    user_audio = r"D:\work\Study\SEM_6\Project\speech_authentication\data\test_user.wav"
    
    if not user_audio:
        print("Error: No audio file provided!")
        sys.exit(1)

    result = authenticate(user_audio)
    print("Authenticated User ID:", result)
