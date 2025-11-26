import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "C:/Users/Surya/Downloads/archive/Data/genres_original"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
    except:
        print("Skipping corrupted:", file_path)
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    return np.hstack([mfcc_mean, centroid, zcr, chroma_mean])


def build_feature_dataset():
    X = []
    y = []

    genres = os.listdir(DATASET_PATH)

    print("Extracting features for all files...")
    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                features = extract_features(file_path)

                if features is None:
                    continue

                X.append(features)
                y.append(genre)

    X = np.array(X)
    y = np.array(y)

    print("Saving to disk...")
    np.save("model_files/X_features.npy", X)
    np.save("model_files/y_labels.npy", y)

    print("Dataset saved successfully!")


if __name__ == "__main__":
    build_feature_dataset()
