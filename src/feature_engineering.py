import os
import sys
import numpy as np
import pandas as pd

def engineer_interaction_features(df):
    """
    Creates advanced domain-specific interaction features, ratios,
    and prosodic dynamics from the base acoustic feature set.
    """
    df = df.copy()

    # 1. Fluency & Temporal Ratios
    df['speech_to_total_ratio'] = df['duration'] * df['speech_ratio'] / (df['duration'] + 1e-6)
    df['pause_frequency_ratio'] = df['pause_count'] / (df['speech_ratio'] + 1e-6)
    df['pause_density_per_sec'] = df['pause_count'] / (df['duration'] + 1e-6)
    df['mean_pause_to_duration_ratio'] = df['pause_mean_duration'] / (df['duration'] + 1e-6)
    df['max_pause_to_mean_pause_ratio'] = df['pause_max_duration'] / (df['pause_mean_duration'] + 1e-6)

    # 2. Energy Dynamics & Coefficient of Variation
    df['rms_cv'] = df['rms_std'] / (df['rms_mean'] + 1e-6) # Energy variability
    df['rms_dynamic_ratio'] = df['rms_max'] / (df['rms_min'] + 1e-6)
    df['rms_iqr_to_range_ratio'] = df['rms_iqr'] / (df['rms_range'] + 1e-6)

    # 3. Pitch (F0) Modulation & Articulation Dynamics
    df['f0_cv'] = df['f0_std'] / (df['f0_mean'] + 1e-6) # Pitch modulation index
    df['f0_range_to_median_ratio'] = df['f0_range'] / (df['f0_median'] + 1e-6)
    df['f0_iqr_to_std_ratio'] = df['f0_iqr'] / (df['f0_std'] + 1e-6)
    df['voiced_density_per_sec'] = df['voiced_frames_count'] / (df['duration'] + 1e-6)

    # 4. Spectral Envelope & Timbre Interactions
    df['spec_spread_ratio'] = df['spec_bw_mean'] / (df['spec_cent_mean'] + 1e-6)
    df['spec_rolloff_to_centroid_ratio'] = df['spec_rolloff_mean'] / (df['spec_cent_mean'] + 1e-6)
    df['spec_flatness_entropy'] = df['spec_flatness_mean'] * np.log(df['spec_flatness_mean'] + 1e-6)

    # 5. High-vs-Low Frequency Contrast Gradients
    if 'spec_contrast_b5_mean' in df.columns and 'spec_contrast_b0_mean' in df.columns:
        df['contrast_gradient_high_low'] = df['spec_contrast_b5_mean'] - df['spec_contrast_b0_mean']
        df['contrast_gradient_mid_low'] = df['spec_contrast_b3_mean'] - df['spec_contrast_b0_mean']

    # 6. MFCC Summary Aggregations (Overall Spectral Energy & Trajectory)
    mfcc_means = [c for c in df.columns if c.startswith('mfcc_') and c.endswith('_mean')]
    mfcc_stds = [c for c in df.columns if c.startswith('mfcc_') and c.endswith('_std')]
    mfcc_ds = [c for c in df.columns if c.startswith('mfcc_d_') and c.endswith('_mean')]
    mfcc_d2s = [c for c in df.columns if c.startswith('mfcc_d2_') and c.endswith('_mean')]

    if len(mfcc_means) > 0:
        df['mfcc_mean_norm'] = np.sqrt(np.sum(df[mfcc_means]**2, axis=1))
        df['mfcc_std_norm'] = np.sqrt(np.sum(df[mfcc_stds]**2, axis=1))
    if len(mfcc_ds) > 0:
        df['mfcc_velocity_energy'] = np.sqrt(np.sum(df[mfcc_ds]**2, axis=1))
    if len(mfcc_d2s) > 0:
        df['mfcc_acceleration_energy'] = np.sqrt(np.sum(df[mfcc_d2s]**2, axis=1))

    # Clean any inf or nan values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def generate_engineered_datasets(train_path, test_path, out_train_path, out_test_path):
    print("Generating advanced engineered feature datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    fe_train = engineer_interaction_features(train_df)
    fe_test = engineer_interaction_features(test_df)

    fe_train.to_csv(out_train_path, index=False)
    fe_test.to_csv(out_test_path, index=False)

    print(f"Saved {len(fe_train)} train samples ({len(fe_train.columns)} cols) to {out_train_path}")
    print(f"Saved {len(fe_test)} test samples ({len(fe_test.columns)} cols) to {out_test_path}")
    return fe_train, fe_test

if __name__ == "__main__":
    feat_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features"
    tr_in = os.path.join(feat_dir, "audio_features_train.csv")
    te_in = os.path.join(feat_dir, "audio_features_test.csv")
    
    tr_out = os.path.join(feat_dir, "engineered_features_train.csv")
    te_out = os.path.join(feat_dir, "engineered_features_test.csv")
    
    generate_engineered_datasets(tr_in, te_in, tr_out, te_out)
