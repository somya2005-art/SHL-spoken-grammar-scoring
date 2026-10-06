import os
import sys
import time
import soundfile as sf
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import scipy.stats as stats
import pandas as pd

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)

def get_mel_filterbank(sr=16000, n_fft=2048, n_mels=40, fmin=0, fmax=8000):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, int(n_fft // 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filterbank

MEL_FB_40 = get_mel_filterbank()

def compute_mfcc_pure_scipy(y, sr=16000, n_fft=2048, hop_length=512, n_mfcc=20):
    # STFT using scipy
    _, _, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, boundary=None)
    mag_spec = np.abs(Zxx)**2
    # Mel filtering
    mel_spec = np.dot(MEL_FB_40, mag_spec)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    log_mel = np.log(mel_spec)
    # DCT-II
    mfcc = fft.dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc]
    return mfcc, np.abs(Zxx)

print("Testing pure scipy MFCC on audio_192.wav...")
data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final\train"
wav_path = os.path.join(data_dir, "audio_192.wav")

y, sr = sf.read(wav_path)
t0 = time.time()
mfcc, spec = compute_mfcc_pure_scipy(y)
t1 = time.time()
print(f"Computed pure scipy MFCC in {t1-t0:.4f}s! MFCC shape: {mfcc.shape}, Spec shape: {spec.shape}")
