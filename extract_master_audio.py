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
data_dir = os.path.join(base_dir, "data", "Dataset_Final")
feat_dir = os.path.join(base_dir, "data", "features")
os.makedirs(feat_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. Mel Filterbank Setup
# -------------------------------------------------------------
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_filterbank(sr=16000, n_fft=2048, n_mels=40, fmin=0, fmax=8000):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
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

# -------------------------------------------------------------
# 2. Fast Autocorrelation Pitch Tracker
# -------------------------------------------------------------
def track_f0_autocorr(y, sr=16000, frame_len=1024, hop=512, fmin=60, fmax=400):
    # Downsample by 2 to 8kHz for fast pitch extraction
    y_ds = y[::2]
    sr_ds = sr // 2
    f_len = frame_len // 2
    h = hop // 2
    
    min_lag = int(sr_ds / fmax)
    max_lag = int(sr_ds / fmin)
    
    n_frames = 1 + int((len(y_ds) - f_len) / h)
    if n_frames <= 0:
        return np.array([])
    
    f0_list = []
    for i in range(0, len(y_ds) - f_len, h * 2): # step by 2 frames for speed
        frame = y_ds[i : i + f_len]
        if np.max(np.abs(frame)) < 1e-4:
            continue
        corr = signal.correlate(frame, frame, mode='full')
        corr = corr[len(frame) - 1 :]
        
        search_region = corr[min_lag : max_lag]
        if len(search_region) == 0:
            continue
        
        peak_idx = np.argmax(search_region) + min_lag
        # Check peak clarity (voicedness)
        if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.35:
            f0 = sr_ds / peak_idx
            if fmin <= f0 <= fmax:
                f0_list.append(f0)
                
    return np.array(f0_list)

# -------------------------------------------------------------
# 3. Master Feature Extractor for Single File
# -------------------------------------------------------------
def extract_master_features(wav_path):
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = signal.resample_poly(y, 16000, sr)
            sr = 16000
        
        y = y.astype(np.float32)
        total_duration = len(y) / sr
        if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            return {'duration': float(total_duration), 'is_silent': 1.0}

        row = {'duration': float(total_duration), 'is_silent': 0.0}

        # --- A. Frame Energy & Dynamics ---
        frame_len = 2048
        hop = 512
        n_frames = 1 + int((len(y) - frame_len) / hop)
        if n_frames > 0:
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
        row['rms_kurtosis'] = float(stats.kurtosis(rms)) if len(rms) > 2 else 0.0
        row['rms_iqr'] = float(stats.iqr(rms)) if len(rms) > 2 else 0.0

        # --- B. Silence, Pauses & Fluency ---
        energy_thresh = max(0.005, float(np.percentile(rms, 25)))
        silent_frames = (rms < energy_thresh)
        silence_ratio = float(np.mean(silent_frames))
        row['silence_ratio'] = silence_ratio
        row['speech_ratio'] = 1.0 - silence_ratio
        
        frame_t = hop / sr
        pauses = []
        cur_p = 0
        for s in silent_frames:
            if s:
                cur_p += 1
            else:
                if cur_p > 0:
                    pauses.append(cur_p * frame_t)
                    cur_p = 0
        if cur_p > 0:
            pauses.append(cur_p * frame_t)

        long_p = [p for p in pauses if p >= 0.25]
        row['pause_count'] = len(long_p)
        row['pause_total_duration'] = float(np.sum(long_p)) if len(long_p) > 0 else 0.0
        row['pause_mean_duration'] = float(np.mean(long_p)) if len(long_p) > 0 else 0.0
        row['pause_max_duration'] = float(np.max(long_p)) if len(long_p) > 0 else 0.0
        row['pause_rate_per_min'] = (len(long_p) / (total_duration / 60.0)) if total_duration > 0 else 0.0
        row['speech_to_pause_ratio'] = float(row['speech_ratio'] / (silence_ratio + 1e-6))

        # --- C. Zero Crossing Rate ---
        zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0
        row['zcr_mean'] = float(zcr)

        # --- D. STFT & Spectral Features ---
        _, _, Zxx = signal.stft(y, fs=sr, nperseg=frame_len, noverlap=frame_len-hop, boundary=None)
        mag = np.abs(Zxx)
        freqs = np.linspace(0, sr/2, mag.shape[0])

        # Centroid & Bandwidth
        spec_sum = np.sum(mag, axis=0) + 1e-12
        centroid = np.sum(freqs[:, None] * mag, axis=0) / spec_sum
        row['spec_cent_mean'] = float(np.mean(centroid))
        row['spec_cent_std'] = float(np.std(centroid))
        row['spec_cent_skew'] = float(stats.skew(centroid)) if len(centroid) > 2 else 0.0

        bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid)**2) * mag, axis=0) / spec_sum)
        row['spec_bw_mean'] = float(np.mean(bandwidth))
        row['spec_bw_std'] = float(np.std(bandwidth))

        # Rolloff (85%)
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

        # --- E. Spectral Contrast (6 octave bands) ---
        bands = [(100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
        for b_idx, (f_low, f_high) in enumerate(bands):
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.sum(band_mask) > 0:
                sub_mag = mag[band_mask, :]
                peak = np.percentile(sub_mag, 95, axis=0)
                valley = np.percentile(sub_mag, 5, axis=0) + 1e-6
                contrast = np.log(peak / valley)
                row[f'spec_contrast_b{b_idx}_mean'] = float(np.mean(contrast))
                row[f'spec_contrast_b{b_idx}_std'] = float(np.std(contrast))
            else:
                row[f'spec_contrast_b{b_idx}_mean'] = 0.0
                row[f'spec_contrast_b{b_idx}_std'] = 0.0

        # --- F. MFCCs (20 cepstral coefficients + deltas + delta-deltas) ---
        mel_spec = np.dot(MEL_FB, mag**2)
        log_mel = np.log(np.maximum(mel_spec, 1e-12))
        mfcc = fft.dct(log_mel, type=2, axis=0, norm='ortho')[:20]
        
        mfcc_d = np.gradient(mfcc, axis=1) if mfcc.shape[1] > 2 else np.zeros_like(mfcc)
        mfcc_d2 = np.gradient(mfcc_d, axis=1) if mfcc_d.shape[1] > 2 else np.zeros_like(mfcc)

        for i in range(20):
            row[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            row[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            row[f'mfcc_{i}_iqr'] = float(stats.iqr(mfcc[i])) if mfcc.shape[1] > 2 else 0.0
            row[f'mfcc_{i}_skew'] = float(stats.skew(mfcc[i])) if mfcc.shape[1] > 2 else 0.0
            row[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_d[i]))
            row[f'mfcc_d_{i}_std'] = float(np.std(mfcc_d[i]))
            row[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_d2[i]))
            row[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_d2[i]))

        # --- G. Pitch (F0) & Voicing Dynamics ---
        f0_arr = track_f0_autocorr(y, sr=sr)
        row['voiced_frames_count'] = len(f0_arr)
        if len(f0_arr) > 5:
            row['f0_mean'] = float(np.mean(f0_arr))
            row['f0_std'] = float(np.std(f0_arr))
            row['f0_median'] = float(np.median(f0_arr))
            row['f0_min'] = float(np.min(f0_arr))
            row['f0_max'] = float(np.max(f0_arr))
            row['f0_range'] = float(np.max(f0_arr) - np.min(f0_arr))
            row['f0_iqr'] = float(stats.iqr(f0_arr))
            # Jitter
            f0_diff = np.abs(np.diff(f0_arr))
            row['f0_jitter_rel'] = float(np.mean(f0_diff) / (np.mean(f0_arr) + 1e-6))
        else:
            for k in ['f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 'f0_range', 'f0_iqr', 'f0_jitter_rel']:
                row[k] = 0.0

        return row

    except Exception as e:
        print(f"Error {wav_path}: {e}", flush=True)
        return {'duration': 0.0, 'is_silent': 1.0}


# -------------------------------------------------------------
# 4. Batch Execution Pipeline
# -------------------------------------------------------------
def run():
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    for name, df, folder, out_name in [
        ("TRAIN", train_df, "train", "audio_features_train.csv"),
        ("TEST", test_df, "test", "audio_features_test.csv")
    ]:
        out_path = os.path.join(feat_dir, out_name)
        total = len(df)
        print(f"\n=======================================================", flush=True)
        print(f"[{name}] Extracting {total} files from {folder}...", flush=True)
        print(f"=======================================================", flush=True)
        
        records = []
        t0 = time.time()
        for idx, row in df.iterrows():
            fn = row['filename']
            wav_p = os.path.join(data_dir, folder, fn)
            feats = extract_master_features(wav_p)
            feats['filename'] = fn
            if 'label' in row:
                feats['label'] = row['label']
            records.append(feats)

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                rem = (total - (idx + 1)) / rate if rate > 0 else 0.0
                pct = ((idx + 1) / total) * 100
                print(f"  [{name}] {idx+1}/{total} ({pct:.1f}%) | {rate:.1f} files/s | Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m", flush=True)

        res_df = pd.DataFrame(records).fillna(0.0)
        cols = ['filename'] + [c for c in res_df.columns if c not in ['filename', 'label']]
        if 'label' in res_df.columns:
            cols.append('label')
        res_df = res_df[cols]
        res_df.to_csv(out_path, index=False)
        total_time = time.time() - t0
        print(f"[{name}] Saved {len(res_df)} rows and {len(res_df.columns)} features -> {out_path} in {total_time/60:.2f} mins.", flush=True)

    print("\nALL MASTER AUDIO FEATURES SUCCESSFULLY EXTRACTED!", flush=True)

if __name__ == "__main__":
    run()
