import os
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def extract_single_audio_features(wav_path, target_sr=16000):
    """
    Extracts rich acoustic, prosodic, spectral, and temporal features from a single audio file.
    """
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        y = y.astype(np.float32)
        total_duration = len(y) / sr

        # If audio is empty or essentially silent
        if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
            return {'duration': float(total_duration), 'is_silent': 1.0}

        features = {
            'duration': float(total_duration),
            'is_silent': 0.0
        }

        # 1. RMS Energy & Dynamics
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_max'] = float(np.max(rms))
        features['rms_min'] = float(np.min(rms))
        features['rms_range'] = float(np.max(rms) - np.min(rms))
        features['rms_skew'] = float(stats.skew(rms)) if len(rms) > 2 else 0.0
        features['rms_kurtosis'] = float(stats.kurtosis(rms)) if len(rms) > 2 else 0.0

        # 2. Pause & Silence Analysis (Fluency metrics)
        energy_threshold = max(0.005, float(np.percentile(rms, 25)))
        silent_frames = rms < energy_threshold
        silence_ratio = float(np.mean(silent_frames))
        features['silence_ratio'] = silence_ratio
        features['speech_ratio'] = 1.0 - silence_ratio
        
        frame_time = 512 / sr
        pause_lengths = []
        cur_pause = 0
        for is_sil in silent_frames:
            if is_sil:
                cur_pause += 1
            else:
                if cur_pause > 0:
                    pause_lengths.append(cur_pause * frame_time)
                    cur_pause = 0
        if cur_pause > 0:
            pause_lengths.append(cur_pause * frame_time)

        long_pauses = [p for p in pause_lengths if p >= 0.25]
        features['pause_count'] = len(long_pauses)
        features['pause_total_duration'] = float(np.sum(long_pauses)) if long_pauses else 0.0
        features['pause_mean_duration'] = float(np.mean(long_pauses)) if long_pauses else 0.0
        features['pause_max_duration'] = float(np.max(long_pauses)) if long_pauses else 0.0
        features['pause_rate_per_min'] = (len(long_pauses) / (total_duration / 60.0)) if total_duration > 0 else 0.0
        features['speech_to_pause_ratio'] = float(features['speech_ratio'] / (silence_ratio + 1e-6))

        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        features['zcr_max'] = float(np.max(zcr))
        features['zcr_skew'] = float(stats.skew(zcr)) if len(zcr) > 2 else 0.0

        # 4. Spectral Features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        features['spec_cent_mean'] = float(np.mean(spec_cent))
        features['spec_cent_std'] = float(np.std(spec_cent))
        features['spec_cent_skew'] = float(stats.skew(spec_cent))

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        features['spec_bw_mean'] = float(np.mean(spec_bw))
        features['spec_bw_std'] = float(np.std(spec_bw))

        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)[0]
        features['spec_rolloff_mean'] = float(np.mean(spec_rolloff))
        features['spec_rolloff_std'] = float(np.std(spec_rolloff))

        spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        features['spec_flatness_mean'] = float(np.mean(spec_flatness))
        features['spec_flatness_std'] = float(np.std(spec_flatness))

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        for band in range(spec_contrast.shape[0]):
            features[f'spec_contrast_b{band}_mean'] = float(np.mean(spec_contrast[band]))
            features[f'spec_contrast_b{band}_std'] = float(np.std(spec_contrast[band]))

        # 5. MFCCs (20 coefficients + deltas + delta-deltas)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        for i in range(20):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            features[f'mfcc_{i}_skew'] = float(stats.skew(mfcc[i]))
            features[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_delta[i]))
            features[f'mfcc_d_{i}_std'] = float(np.std(mfcc_delta[i]))
            features[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
            features[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_delta2[i]))

        # 6. Pitch (F0) Dynamics via fast downsampled YIN
        y_pitch = y[::4] # downsample to 4kHz
        f0 = librosa.yin(y_pitch, fmin=60, fmax=400, sr=4000, hop_length=128)
        valid_f0 = f0[(f0 >= 60) & (f0 <= 400) & (~np.isnan(f0))]
        voiced_ratio = len(valid_f0) / (len(f0) + 1e-6)
        features['voiced_ratio'] = float(voiced_ratio)

        if len(valid_f0) > 10:
            features['f0_mean'] = float(np.mean(valid_f0))
            features['f0_std'] = float(np.std(valid_f0))
            features['f0_median'] = float(np.median(valid_f0))
            features['f0_min'] = float(np.min(valid_f0))
            features['f0_max'] = float(np.max(valid_f0))
            features['f0_range'] = float(np.max(valid_f0) - np.min(valid_f0))
            features['f0_p10'] = float(np.percentile(valid_f0, 10))
            features['f0_p90'] = float(np.percentile(valid_f0, 90))
            features['f0_iqr'] = float(stats.iqr(valid_f0))
            features['f0_skew'] = float(stats.skew(valid_f0))
            diff_f0 = np.abs(np.diff(valid_f0))
            features['f0_jitter_rel'] = float(np.mean(diff_f0) / (np.mean(valid_f0) + 1e-6))
        else:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_median'] = 0.0
            features['f0_min'] = 0.0
            features['f0_max'] = 0.0
            features['f0_range'] = 0.0
            features['f0_p10'] = 0.0
            features['f0_p90'] = 0.0
            features['f0_iqr'] = 0.0
            features['f0_skew'] = 0.0
            features['f0_jitter_rel'] = 0.0

        # 7. Rhythm, Onset & Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        features['onset_mean'] = float(np.mean(onset_env))
        features['onset_std'] = float(np.std(onset_env))
        features['onset_max'] = float(np.max(onset_env))
        try:
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
            features['tempo_bpm'] = float(tempo[0]) if len(tempo) > 0 else 0.0
        except Exception:
            features['tempo_bpm'] = 0.0

        # 8. Chroma features (harmonic stability)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        for c in range(12):
            features[f'chroma_{c}_mean'] = float(np.mean(chroma[c]))
            features[f'chroma_{c}_std'] = float(np.std(chroma[c]))

        return features

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return {'duration': 0.0, 'is_silent': 1.0}


def _worker_extract(args):
    wav_path, fn, label = args
    feats = extract_single_audio_features(wav_path)
    feats['filename'] = fn
    if label is not None:
        feats['label'] = label
    return feats


def extract_all_audio_features(df, audio_dir, output_csv, max_workers=6):
    """
    Extracts features for all audio files in parallel across threads and saves to CSV.
    """
    tasks = []
    for _, row in df.iterrows():
        fn = row['filename']
        label = row['label'] if 'label' in row else None
        wav_path = os.path.join(audio_dir, fn)
        tasks.append((wav_path, fn, label))

    print(f"Extracting features for {len(tasks)} files using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_features = list(tqdm(executor.map(_worker_extract, tasks), total=len(tasks), desc=f"Extracting ({os.path.basename(audio_dir)})"))

    features_df = pd.DataFrame(all_features)
    cols = ['filename'] + [c for c in features_df.columns if c not in ['filename', 'label']]
    if 'label' in features_df.columns:
        cols.append('label')
    features_df = features_df[cols].fillna(0.0)
    features_df.to_csv(output_csv, index=False)
    print(f"Saved {len(features_df)} rows and {len(features_df.columns)} features to {output_csv}")
    return features_df


if __name__ == "__main__":
    data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    out_train_csv = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\audio_features_train.csv"
    out_test_csv = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\audio_features_test.csv"
    
    print("Extracting Train Audio Features...")
    extract_all_audio_features(train_df, os.path.join(data_dir, "train"), out_train_csv, max_workers=6)
    
    print("Extracting Test Audio Features...")
    extract_all_audio_features(test_df, os.path.join(data_dir, "test"), out_test_csv, max_workers=6)
