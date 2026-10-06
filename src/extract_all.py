import os
import sys
import time
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. Acoustic & Prosodic Feature Extraction
# -------------------------------------------------------------
def extract_single_audio(wav_path, target_sr=16000):
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

        # Energy & RMS
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_std'] = float(np.std(rms))
        feats['rms_max'] = float(np.max(rms))
        feats['rms_min'] = float(np.min(rms))
        feats['rms_range'] = float(np.max(rms) - np.min(rms))
        feats['rms_skew'] = float(stats.skew(rms)) if len(rms) > 2 else 0.0

        # Silence & Pauses
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

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        feats['zcr_mean'] = float(np.mean(zcr))
        feats['zcr_std'] = float(np.std(zcr))
        feats['zcr_max'] = float(np.max(zcr))

        # Spectral
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        feats['spec_cent_mean'] = float(np.mean(sc))
        feats['spec_cent_std'] = float(np.std(sc))

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

        # MFCCs + deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        for i in range(20):
            feats[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            feats[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            feats[f'mfcc_d_{i}_mean'] = float(np.mean(mfcc_d[i]))
            feats[f'mfcc_d_{i}_std'] = float(np.std(mfcc_d[i]))
            feats[f'mfcc_d2_{i}_mean'] = float(np.mean(mfcc_d2[i]))
            feats[f'mfcc_d2_{i}_std'] = float(np.std(mfcc_d2[i]))

        # Pitch F0 via downsampled YIN
        y_pitch = y[::4]
        f0 = librosa.yin(y_pitch, fmin=60, fmax=400, sr=4000, hop_length=128)
        valid_f0 = f0[(f0 >= 60) & (f0 <= 400) & (~np.isnan(f0))]
        feats['voiced_ratio'] = float(len(valid_f0) / (len(f0) + 1e-6))
        if len(valid_f0) > 10:
            feats['f0_mean'] = float(np.mean(valid_f0))
            feats['f0_std'] = float(np.std(valid_f0))
            feats['f0_median'] = float(np.median(valid_f0))
            feats['f0_range'] = float(np.max(valid_f0) - np.min(valid_f0))
            feats['f0_p10'] = float(np.percentile(valid_f0, 10))
            feats['f0_p90'] = float(np.percentile(valid_f0, 90))
            feats['f0_iqr'] = float(stats.iqr(valid_f0))
            diff_f0 = np.abs(np.diff(valid_f0))
            feats['f0_jitter_rel'] = float(np.mean(diff_f0) / (np.mean(valid_f0) + 1e-6))
        else:
            for k in ['f0_mean', 'f0_std', 'f0_median', 'f0_range', 'f0_p10', 'f0_p90', 'f0_iqr', 'f0_jitter_rel']:
                feats[k] = 0.0

        # Onset & Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        feats['onset_mean'] = float(np.mean(onset_env))
        feats['onset_std'] = float(np.std(onset_env))
        try:
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
            feats['tempo_bpm'] = float(tempo[0]) if len(tempo) > 0 else 0.0
        except Exception:
            feats['tempo_bpm'] = 0.0

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        for c in range(12):
            feats[f'chroma_{c}_mean'] = float(np.mean(chroma[c]))
            feats[f'chroma_{c}_std'] = float(np.std(chroma[c]))

        return feats
    except Exception as e:
        print(f"Error {wav_path}: {e}")
        return {'duration': 0.0, 'is_silent': 1.0}


def process_audio_set(df, audio_dir, output_csv):
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        if len(existing) == len(df):
            print(f"Audio features already extracted ({len(existing)} records) at {output_csv}")
            return existing

    results = []
    # Check if partial checkpoint exists
    if os.path.exists(output_csv + ".partial"):
        try:
            results = pd.read_csv(output_csv + ".partial").to_dict('records')
            processed_fns = set(r['filename'] for r in results)
            print(f"Resuming audio extraction from checkpoint with {len(results)} records...")
        except Exception:
            results = []
            processed_fns = set()
    else:
        processed_fns = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Audio Features ({os.path.basename(audio_dir)})"):
        fn = row['filename']
        if fn in processed_fns:
            continue
        wav_p = os.path.join(audio_dir, fn)
        f_dict = extract_single_audio(wav_p)
        f_dict['filename'] = fn
        if 'label' in row:
            f_dict['label'] = row['label']
        results.append(f_dict)
        processed_fns.add(fn)

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(output_csv + ".partial", index=False)

    res_df = pd.DataFrame(results).fillna(0.0)
    cols = ['filename'] + [c for c in res_df.columns if c not in ['filename', 'label']]
    if 'label' in res_df.columns:
        cols.append('label')
    res_df = res_df[cols]
    res_df.to_csv(output_csv, index=False)
    if os.path.exists(output_csv + ".partial"):
        os.remove(output_csv + ".partial")
    print(f"Saved {len(res_df)} audio feature records to {output_csv}")
    return res_df


# -------------------------------------------------------------
# 2. Whisper ASR Transcription
# -------------------------------------------------------------
def process_transcription_set(df, audio_dir, output_csv, pipe):
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        if len(existing) == len(df):
            print(f"Transcripts already exist ({len(existing)} records) at {output_csv}")
            return existing

    results = []
    if os.path.exists(output_csv + ".partial"):
        try:
            results = pd.read_csv(output_csv + ".partial").to_dict('records')
            processed_fns = set(r['filename'] for r in results)
            print(f"Resuming transcription from checkpoint with {len(results)} records...")
        except Exception:
            results = []
            processed_fns = set()
    else:
        processed_fns = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Transcribing ({os.path.basename(audio_dir)})"):
        fn = row['filename']
        if fn in processed_fns:
            continue
        wav_p = os.path.join(audio_dir, fn)
        try:
            y, sr = sf.read(wav_p)
            if y.ndim > 1: y = np.mean(y, axis=1)
            res = pipe({"array": y.astype(np.float32), "sampling_rate": sr})
            txt = res.get("text", "").strip()
        except Exception as e:
            print(f"ASR Error {fn}: {e}")
            txt = ""

        rec = {
            'filename': fn,
            'transcript': txt,
            'char_length': len(txt),
            'word_count': len(txt.split())
        }
        if 'label' in row:
            rec['label'] = row['label']
        results.append(rec)
        processed_fns.add(fn)

        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv + ".partial", index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    if os.path.exists(output_csv + ".partial"):
        os.remove(output_csv + ".partial")
    print(f"Saved {len(res_df)} transcripts to {output_csv}")
    return res_df


# -------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------
if __name__ == "__main__":
    base_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring"
    data_dir = os.path.join(base_dir, "data", "Dataset_Final")
    feat_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(feat_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # 1. Audio Features
    print("\n>>> STEP 1: Acoustic & Prosodic Feature Extraction")
    process_audio_set(train_df, os.path.join(data_dir, "train"), os.path.join(feat_dir, "audio_features_train.csv"))
    process_audio_set(test_df, os.path.join(data_dir, "test"), os.path.join(feat_dir, "audio_features_test.csv"))

    # 2. Whisper Transcriptions
    print("\n>>> STEP 2: Speech-to-Text Transcription with Whisper")
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu", chunk_length_s=30)
    process_transcription_set(train_df, os.path.join(data_dir, "train"), os.path.join(feat_dir, "transcripts_train.csv"), pipe)
    process_transcription_set(test_df, os.path.join(data_dir, "test"), os.path.join(feat_dir, "transcripts_test.csv"), pipe)

    # 3. Text & Linguistic Features
    print("\n>>> STEP 3: NLP & Grammar Feature Extraction")
    from src.text_features import extract_all_text_features
    extract_all_text_features(
        os.path.join(feat_dir, "transcripts_train.csv"),
        os.path.join(feat_dir, "transcripts_test.csv"),
        os.path.join(feat_dir, "text_features_train.csv"),
        os.path.join(feat_dir, "text_features_test.csv"),
        n_tfidf_components=16
    )

    print("\n>>> ALL MULTIMODAL FEATURES EXTRACTED SUCCESSFULLY!")
