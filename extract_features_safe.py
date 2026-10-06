import os
import sys
import time
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import pandas as pd

base_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring"
data_dir = os.path.join(base_dir, "data", "Dataset_Final")
feat_dir = os.path.join(base_dir, "data", "features")
os.makedirs(feat_dir, exist_ok=True)

def extract_features(wav_path):
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        y = y.astype(np.float32)
        total_duration = len(y) / sr
        if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            return {'duration': float(total_duration), 'is_silent': 1.0}

        row = {'duration': float(total_duration), 'is_silent': 0.0}

        # 1. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        row['rms_mean'] = float(np.mean(rms))
        row['rms_std'] = float(np.std(rms))
        row['rms_max'] = float(np.max(rms))
        row['rms_min'] = float(np.min(rms))
        row['rms_range'] = float(np.max(rms) - np.min(rms))
        row['rms_skew'] = float(stats.skew(rms)) if len(rms) > 2 else 0.0
        row['rms_kurtosis'] = float(stats.kurtosis(rms)) if len(rms) > 2 else 0.0
        row['rms_iqr'] = float(stats.iqr(rms)) if len(rms) > 2 else 0.0

        # 2. Silence, Pauses & Fluency
        energy_thresh = max(0.005, float(np.percentile(rms, 25)))
        silent_frames = (rms < energy_thresh)
        silence_ratio = float(np.mean(silent_frames))
        row['silence_ratio'] = silence_ratio
        row['speech_ratio'] = 1.0 - silence_ratio
        
        frame_t = 512 / sr
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

        # 3. ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        row['zcr_mean'] = float(np.mean(zcr))
        row['zcr_std'] = float(np.std(zcr))
        row['zcr_max'] = float(np.max(zcr))
        row['zcr_skew'] = float(stats.skew(zcr)) if len(zcr) > 2 else 0.0

        # 4. Spectral Centroid, Bandwidth, Rolloff, Flatness
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        row['spec_cent_mean'] = float(np.mean(sc))
        row['spec_cent_std'] = float(np.std(sc))
        row['spec_cent_skew'] = float(stats.skew(sc)) if len(sc) > 2 else 0.0

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        row['spec_bw_mean'] = float(np.mean(sb))
        row['spec_bw_std'] = float(np.std(sb))

        sr_ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)[0]
        row['spec_rolloff_mean'] = float(np.mean(sr_ro))
        row['spec_rolloff_std'] = float(np.std(sr_ro))

        sf_fl = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        row['spec_flatness_mean'] = float(np.mean(sf_fl))
        row['spec_flatness_std'] = float(np.std(sf_fl))

        # 5. Spectral Contrast (7 frequency bands)
        sc_ct = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        for b in range(sc_ct.shape[0]):
            row[f'spec_contrast_b{b}_mean'] = float(np.mean(sc_ct[b]))
            row[f'spec_contrast_b{b}_std'] = float(np.std(sc_ct[b]))

        # 6. MFCCs (1..20 + deltas + delta-deltas = 60 streams)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        for i in range(20):
            row[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            row[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            row[f'mfcc_{i}_iqr'] = float(stats.iqr(mfcc[i])) if len(mfcc[i]) > 2 else 0.0
            row[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_d[i]))
            row[f'mfcc_d_{i}_std'] = float(np.std(mfcc_d[i]))
            row[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_d2[i]))
            row[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_d2[i]))

        # 7. Chroma STFT (12 harmonic pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        for c in range(12):
            row[f'chroma_{c}_mean'] = float(np.mean(chroma[c]))
            row[f'chroma_{c}_std'] = float(np.std(chroma[c]))

        return row
    except Exception as e:
        print(f"Error {wav_path}: {e}", flush=True)
        return {'duration': 0.0, 'is_silent': 1.0}


def run_extraction():
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
            feats = extract_features(wav_p)
            feats['filename'] = fn
            if 'label' in row:
                feats['label'] = row['label']
            records.append(feats)

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                rem = (total - (idx + 1)) / rate if rate > 0 else 0.0
                pct = ((idx + 1) / total) * 100
                print(f"  [{name}] {idx+1}/{total} ({pct:.1f}%) | Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m", flush=True)

        res_df = pd.DataFrame(records).fillna(0.0)
        cols = ['filename'] + [c for c in res_df.columns if c not in ['filename', 'label']]
        if 'label' in res_df.columns:
            cols.append('label')
        res_df = res_df[cols]
        res_df.to_csv(out_path, index=False)
        total_time = time.time() - t0
        print(f"[{name}] Successfully saved {len(res_df)} rows and {len(res_df.columns)} features to {out_path} in {total_time/60:.2f} mins.", flush=True)

    print("\nALL FEATURE EXTRACTION COMPLETE!", flush=True)

if __name__ == "__main__":
    run_extraction()
