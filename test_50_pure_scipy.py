import os
import sys
import time
import soundfile as sf
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import scipy.stats as stats
import pandas as pd

base_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring"
data_dir = os.path.join(base_dir, "data", "Dataset_Final", "train")
train_df = pd.read_csv(os.path.join(base_dir, "data", "Dataset_Final", "train.csv"))

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

MEL_FB = get_mel_filterbank()

def extract_features(wav_p):
    y, sr = sf.read(wav_p)
    if y.ndim > 1: y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    duration = len(y) / sr
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        return {'duration': duration, 'is_silent': 1.0}

    row = {'duration': duration, 'is_silent': 0.0}

    # Frame energy (2048 window, 512 hop)
    frame_len = 2048
    hop = 512
    n_frames = 1 + int((len(y) - frame_len) / hop)
    if n_frames > 0:
        # Vectorized framing
        shape = (n_frames, frame_len)
        strides = (y.strides[0] * hop, y.strides[0])
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    else:
        rms = np.array([np.sqrt(np.mean(y**2) + 1e-12)])

    row['rms_mean'] = float(np.mean(rms))
    row['rms_std'] = float(np.std(rms))
    row['rms_max'] = float(np.max(rms))
    row['rms_min'] = float(np.min(rms))
    row['rms_range'] = float(np.max(rms) - np.min(rms))
    row['rms_skew'] = float(stats.skew(rms)) if len(rms) > 2 else 0.0
    row['rms_iqr'] = float(stats.iqr(rms)) if len(rms) > 2 else 0.0

    # Silence & Pause fluency
    energy_thresh = max(0.005, float(np.percentile(rms, 25)))
    sil = rms < energy_thresh
    sil_ratio = float(np.mean(sil))
    row['silence_ratio'] = sil_ratio
    row['speech_ratio'] = 1.0 - sil_ratio

    frame_t = hop / sr
    pauses = []
    cp = 0
    for s in sil:
        if s: cp += 1
        else:
            if cp > 0: pauses.append(cp * frame_t); cp = 0
    if cp > 0: pauses.append(cp * frame_t)

    lp = [p for p in pauses if p >= 0.25]
    row['pause_count'] = len(lp)
    row['pause_total_duration'] = float(np.sum(lp)) if len(lp) > 0 else 0.0
    row['pause_mean_duration'] = float(np.mean(lp)) if len(lp) > 0 else 0.0
    row['pause_max_duration'] = float(np.max(lp)) if len(lp) > 0 else 0.0
    row['pause_rate_per_min'] = (len(lp) / (duration / 60.0)) if duration > 0 else 0.0
    row['speech_to_pause_ratio'] = float(row['speech_ratio'] / (sil_ratio + 1e-6))

    # Zero Crossing Rate
    zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0
    row['zcr_mean'] = float(zcr)

    # STFT & Spectral
    _, _, Zxx = signal.stft(y, fs=sr, nperseg=frame_len, noverlap=frame_len-hop, boundary=None)
    mag = np.abs(Zxx)
    freqs = np.linspace(0, sr/2, mag.shape[0])

    # Spectral Centroid
    spec_sum = np.sum(mag, axis=0) + 1e-12
    centroid = np.sum(freqs[:, None] * mag, axis=0) / spec_sum
    row['spec_cent_mean'] = float(np.mean(centroid))
    row['spec_cent_std'] = float(np.std(centroid))
    row['spec_cent_skew'] = float(stats.skew(centroid)) if len(centroid) > 2 else 0.0

    # Spectral Bandwidth
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid)**2) * mag, axis=0) / spec_sum)
    row['spec_bw_mean'] = float(np.mean(bandwidth))
    row['spec_bw_std'] = float(np.std(bandwidth))

    # Spectral Rolloff (85%)
    cum_mag = np.cumsum(mag, axis=0)
    rolloff_idx = np.apply_along_axis(lambda col: np.searchsorted(col, 0.85 * col[-1]), axis=0, arr=cum_mag)
    rolloff_freq = freqs[rolloff_idx]
    row['spec_rolloff_mean'] = float(np.mean(rolloff_freq))
    row['spec_rolloff_std'] = float(np.std(rolloff_freq))

    # Spectral Flatness
    geom_mean = np.exp(np.mean(np.log(mag + 1e-12), axis=0))
    arith_mean = np.mean(mag, axis=0) + 1e-12
    flatness = geom_mean / arith_mean
    row['spec_flatness_mean'] = float(np.mean(flatness))
    row['spec_flatness_std'] = float(np.std(flatness))

    # MFCCs (1..20 + deltas + delta2)
    mel_spec = np.dot(MEL_FB, mag**2)
    log_mel = np.log(np.maximum(mel_spec, 1e-12))
    mfcc = fft.dct(log_mel, type=2, axis=0, norm='ortho')[:20]
    
    # Deltas
    mfcc_d = np.gradient(mfcc, axis=1)
    mfcc_d2 = np.gradient(mfcc_d, axis=1)

    for i in range(20):
        row[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
        row[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
        row[f'mfcc_{i}_iqr'] = float(stats.iqr(mfcc[i])) if mfcc.shape[1] > 2 else 0.0
        row[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_d[i]))
        row[f'mfcc_d_{i}_std'] = float(np.std(mfcc_d[i]))
        row[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_d2[i]))
        row[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_d2[i]))

    return row

t0 = time.time()
for idx, row in train_df.iloc[:50].iterrows():
    fn = row['filename']
    f_dict = extract_features(os.path.join(data_dir, fn))
t1 = time.time()
print(f"50 files extracted in {t1-t0:.2f}s! ({len(f_dict)} features per file). Rate: {50/(t1-t0):.1f} files/sec!")
