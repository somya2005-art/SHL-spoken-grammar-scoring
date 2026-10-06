import soundfile as sf
import numpy as np
import librosa
import time

wav_path = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final\train\audio_192.wav"
y, sr = sf.read(wav_path)
print(f"Loaded audio shape: {y.shape}, sr: {sr}")

# Test pitch with autocorrelation / downsampled YIN
t0 = time.time()
y_sub = y[::4] # downsample to 4000Hz (human voice F0 fundamental is 60-400Hz, Nyquist 2000Hz is plenty)
f0 = librosa.yin(y_sub, fmin=60, fmax=400, sr=4000, hop_length=128)
t1 = time.time()
print(f"Pitch extraction in {t1-t0:.3f}s: mean F0 = {np.nanmean(f0):.1f} Hz")

# Test MFCC
t0 = time.time()
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
t1 = time.time()
print(f"MFCC in {t1-t0:.3f}s, shape: {mfcc.shape}")

# Test Spectral
t0 = time.time()
cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
t1 = time.time()
print(f"Spectral centroid in {t1-t0:.3f}s")
