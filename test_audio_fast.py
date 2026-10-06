import time
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import os

data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final\train"
wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)[:10]]

t0 = time.time()
for wav_p in wav_files:
    y, sr = sf.read(wav_p)
    if y.ndim > 1: y = np.mean(y, axis=1)
    
    # 1. Energy & Pauses
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
    
    # 2. Spectral (centroid, bandwidth, rolloff, flatness, contrast)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
    s_ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    s_fl = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
    s_ct = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
    
    # 3. MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    
    # 4. Chroma & Tempo
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)

t1 = time.time()
print(f"10 audio files processed in {t1-t0:.2f}s (avg: {(t1-t0)/10:.3f}s per 55s file)")
