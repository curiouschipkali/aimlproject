import joblib
import librosa
import numpy as np

knn = joblib.load("model_files/model_knn.pkl")
svm = joblib.load("model_files/model_svm.pkl")
rf  = joblib.load("model_files/model_rf.pkl")
gbt = joblib.load("model_files/model_gbt.pkl")

scaler = joblib.load("model_files/scaler.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    return np.hstack([mfcc_mean, centroid, zcr, chroma_mean]).reshape(1, -1)

file = "C:/Users/Surya/Downloads/archive/Data/genres_original/pop/pop.00045.wav"


features = extract_features(file)
features_scaled = scaler.transform(features)

print("\n--- Predictions ---")

print("KNN:", knn.predict(features_scaled)[0])
print("SVM:", svm.predict(features_scaled)[0])
print("RF:",  rf.predict(features)[0])
print("GBT:", gbt.predict(features)[0])
