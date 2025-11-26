# predict_file.py

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from typing import Annotated

from fastapi.middleware.cors import CORSMiddleware

# Load trained model + scaler
# model = joblib.load("model_files/model_knn.pkl")
# scaler = joblib.load("model_files/scaler.pkl")

ALLOWED_ORIGINS = [
    "http://localhost",
    "*",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    return np.hstack([mfcc_mean, centroid, zcr, chroma_mean]).reshape(1, -1)


# file_path = "audio.wav"

# features = extract_features(file_path)
# features_scaled = scaler.transform(features)

# prediction = model.predict(features_scaled)[0]
# print("\nPredicted Genre:", prediction)



@app.post('/api/testaudio')
async def test_audio(audio_file: Annotated[UploadFile, File()],model_type: Annotated[str, Form(...)]):
    with open("audio.wav", "wb") as f:
        content = await audio_file.read()
        f.write(content)
    
    print(f"loading model {model_type}")

    model = joblib.load(f"model_files/model_{model_type}.pkl")
    scaler = joblib.load(f"model_files/scaler.pkl")
    
    features = extract_features("audio.wav")
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    print("\nPredicted Genre:", prediction)
    
    return {
        "results":prediction
    }
    

@app.get("/")
def hello():
    return {"message": "Hello, World!"}