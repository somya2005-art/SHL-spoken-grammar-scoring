import os
import sys
import traceback
import soundfile as sf
import numpy as np
import librosa
import scipy.stats as stats
import pandas as pd

data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

print(f"Total train rows: {len(train_df)}")

for idx, row in train_df.iloc[:5].iterrows():
    fn = row['filename']
    wav_p = os.path.join(data_dir, "train", fn)
    print(f"\nProcessing {idx}: {fn}")
    try:
        y, sr = sf.read(wav_p)
        if y.ndim > 1: y = np.mean(y, axis=1)
        print(f"  Loaded: shape={y.shape}, sr={sr}")
        
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        print(f"  RMS computed: {np.mean(rms):.4f}")
        
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)[0]
        print(f"  ZCR computed: {np.mean(zcr):.4f}")
        
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        print(f"  Centroid computed: {np.mean(sc):.4f}")
        
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        print(f"  Bandwidth computed: {np.mean(sb):.4f}")
        
        sr_ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)[0]
        print(f"  Rolloff computed: {np.mean(sr_ro):.4f}")
        
        sf_fl = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        print(f"  Flatness computed: {np.mean(sf_fl):.4f}")
        
        sc_ct = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        print(f"  Contrast computed: shape={sc_ct.shape}")
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        print(f"  MFCC computed: shape={mfcc.shape}")
        
        mfcc_d = librosa.feature.delta(mfcc)
        mfcc_d2 = librosa.feature.delta(mfcc, order=2)
        print(f"  MFCC deltas computed")
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        print(f"  Onset env computed")
        
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=512)
        print(f"  Tempo computed: {tempo}")
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        print(f"  Chroma computed: shape={chroma.shape}")
        
        print(f"  SUCCESS for {fn}!")
    except Exception as e:
        print(f"  ERROR for {fn}: {e}")
        traceback.print_exc()
