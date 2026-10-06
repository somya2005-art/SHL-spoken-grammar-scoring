import sys
import os
import traceback
import time
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats

print("Python:", sys.version)
data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final"
wav_path = os.path.join(data_dir, "train", "audio_192.wav")

try:
    print("Reading WAV...")
    y, sr = sf.read(wav_path)
    print("Shape:", y.shape, "SR:", sr)

    print("Computing RMS...")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    print("RMS mean:", np.mean(rms))

    print("Computing ZCR...")
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
    print("ZCR mean:", np.mean(zcr))

    print("Computing Centroid...")
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    print("SC mean:", np.mean(sc))

    print("Computing MFCC...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
    print("MFCC shape:", mfcc.shape)

    print("Computing Chroma...")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    print("Chroma shape:", chroma.shape)

    print("Computing Onset & Tempo...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
    print("Tempo:", tempo)

    print("Computing Pitch...")
    y_pitch = y[::4]
    f0 = librosa.yin(y_pitch, fmin=60, fmax=400, sr=4000, hop_length=128)
    print("F0 mean:", np.nanmean(f0))

    print("ALL TESTS PASSED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
