import os
import sys
import time
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def extract_comprehensive_audio(wav_path, target_sr=16000):
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        y = y.astype(np.float32)
        total_duration = len(y) / sr

        if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            return {'duration': float(total_duration), 'is_silent': 1.0}

        feats = {'duration': float(total_duration), 'is_silent': 0.0}

        # 1. Energy & RMS Dynamics
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_std'] = float(np.std(rms))
        feats['rms_max'] = float(np.max(rms))
        feats['rms_min'] = float(np.min(rms))
        feats['rms_range'] = float(np.max(rms) - np.min(rms))
        feats['rms_skew'] = float(stats.skew(rms)) if len(rms) > 2 else 0.0
        feats['rms_kurtosis'] = float(stats.kurtosis(rms)) if len(rms) > 2 else 0.0
        feats['rms_iqr'] = float(stats.iqr(rms)) if len(rms) > 2 else 0.0

        # 2. Silence, Pauses & Speech Fluency
        energy_thresh = max(0.005, float(np.percentile(rms, 25)))
        silent_frames = rms < energy_thresh
        silence_ratio = float(np.mean(silent_frames))
        feats['silence_ratio'] = silence_ratio
        feats['speech_ratio'] = 1.0 - silence_ratio
        
        frame_t = 512 / sr
        pauses = []
        cur_p = 0
        for s in silent_frames:
            if s: cur_p += 1
            else:
                if cur_p > 0:
                    pauses.append(cur_p * frame_t)
                    cur_p = 0
        if cur_p > 0: pauses.append(cur_p * frame_t)

        long_p = [p for p in pauses if p >= 0.25]
        feats['pause_count'] = len(long_p)
        feats['pause_total_duration'] = float(np.sum(long_p)) if long_p else 0.0
        feats['pause_mean_duration'] = float(np.mean(long_p)) if long_p else 0.0
        feats['pause_max_duration'] = float(np.max(long_p)) if long_p else 0.0
        feats['pause_rate_per_min'] = (len(long_p) / (total_duration / 60.0)) if total_duration > 0 else 0.0
        feats['speech_to_pause_ratio'] = float(feats['speech_ratio'] / (silence_ratio + 1e-6))

        # 3. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        feats['zcr_mean'] = float(np.mean(zcr))
        feats['zcr_std'] = float(np.std(zcr))
        feats['zcr_max'] = float(np.max(zcr))
        feats['zcr_skew'] = float(stats.skew(zcr)) if len(zcr) > 2 else 0.0

        # 4. Spectral Features (Centroid, Bandwidth, Rolloff, Flatness, Contrast)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        feats['spec_cent_mean'] = float(np.mean(sc))
        feats['spec_cent_std'] = float(np.std(sc))
        feats['spec_cent_skew'] = float(stats.skew(sc)) if len(sc) > 2 else 0.0

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        feats['spec_bw_mean'] = float(np.mean(sb))
        feats['spec_bw_std'] = float(np.std(sb))

        sr_ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)[0]
        feats['spec_rolloff_mean'] = float(np.mean(sr_ro))
        feats['spec_rolloff_std'] = float(np.std(sr_ro))

        sf_fl = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        feats['spec_flatness_mean'] = float(np.mean(sf_fl))
        feats['spec_flatness_std'] = float(np.std(sf_fl))

        sc_ct = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        for b in range(sc_ct.shape[0]):
            feats[f'spec_contrast_b{b}_mean'] = float(np.mean(sc_ct[b]))
            feats[f'spec_contrast_b{b}_std'] = float(np.std(sc_ct[b]))

        # 5. MFCCs (1..20 + deltas + delta-deltas = 60 streams)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        for i in range(20):
            feats[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            feats[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            feats[f'mfcc_{i}_skew'] = float(stats.skew(mfcc[i])) if len(mfcc[i]) > 2 else 0.0
            feats[f'mfcc_{i}_iqr'] = float(stats.iqr(mfcc[i])) if len(mfcc[i]) > 2 else 0.0
            feats[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_d[i]))
            feats[f'mfcc_d_{i}_std'] = float(np.std(mfcc_d[i]))
            feats[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_d2[i]))
            feats[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_d2[i]))

        # 6. Rhythm, Onset Strength & Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        feats['onset_mean'] = float(np.mean(onset_env))
        feats['onset_std'] = float(np.std(onset_env))
        feats['onset_max'] = float(np.max(onset_env))
        feats['onset_skew'] = float(stats.skew(onset_env)) if len(onset_env) > 2 else 0.0
        try:
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
            feats['tempo_bpm'] = float(tempo[0]) if len(tempo) > 0 else 0.0
        except Exception:
            feats['tempo_bpm'] = 0.0

        # 7. Chroma Features (12 harmonic pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        for c in range(12):
            feats[f'chroma_{c}_mean'] = float(np.mean(chroma[c]))
            feats[f'chroma_{c}_std'] = float(np.std(chroma[c]))

        return feats
    except Exception as e:
        print(f"Error {wav_path}: {e}")
        return {'duration': 0.0, 'is_silent': 1.0}


def process_features(df, audio_dir, output_csv):
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        if len(existing) == len(df) and len(existing.columns) > 100:
            print(f"Features already extracted ({len(existing)} rows, {len(existing.columns)} cols) at {output_csv}")
            return existing

    results = []
    checkpoint_file = output_csv + ".chk.csv"
    processed_fns = set()
    if os.path.exists(checkpoint_file):
        try:
            results = pd.read_csv(checkpoint_file).to_dict('records')
            processed_fns = set(r['filename'] for r in results)
            print(f"Resuming {os.path.basename(audio_dir)} from checkpoint ({len(results)} done)...")
        except Exception:
            results = []
            processed_fns = set()

    t_start = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting Audio ({os.path.basename(audio_dir)})"):
        fn = row['filename']
        if fn in processed_fns:
            continue
        wav_p = os.path.join(audio_dir, fn)
        f_dict = extract_comprehensive_audio(wav_p)
        f_dict['filename'] = fn
        if 'label' in row:
            f_dict['label'] = row['label']
        results.append(f_dict)
        processed_fns.add(fn)

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)

    res_df = pd.DataFrame(results).fillna(0.0)
    cols = ['filename'] + [c for c in res_df.columns if c not in ['filename', 'label']]
    if 'label' in res_df.columns:
        cols.append('label')
    res_df = res_df[cols]
    res_df.to_csv(output_csv, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    t_elapsed = time.time() - t_start
    print(f"Extracted {len(res_df)} files ({len(res_df.columns)} features) in {t_elapsed:.1f}s -> Saved to {output_csv}")
    return res_df


if __name__ == "__main__":
    base_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring"
    data_dir = os.path.join(base_dir, "data", "Dataset_Final")
    feat_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(feat_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    out_train_csv = os.path.join(feat_dir, "audio_features_train.csv")
    out_test_csv = os.path.join(feat_dir, "audio_features_test.csv")

    print("\n>>> Extracting Train Audio Features...")
    process_features(train_df, os.path.join(data_dir, "train"), out_train_csv)

    print("\n>>> Extracting Test Audio Features...")
    process_features(test_df, os.path.join(data_dir, "test"), out_test_csv)

    print("\n>>> AUDIO FEATURE EXTRACTION COMPLETE!")
