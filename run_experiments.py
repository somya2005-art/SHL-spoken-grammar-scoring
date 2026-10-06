import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train_and_evaluate import run_experiment_pipeline

if __name__ == "__main__":
    feat_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features"
    
    print("\n=======================================================")
    print(">>> EXPERIMENT 1: Acoustic Baseline (209 features)")
    print("=======================================================")
    run_experiment_pipeline(
        os.path.join(feat_dir, "audio_features_train.csv"),
        os.path.join(feat_dir, "audio_features_test.csv"),
        exp_name="v1_acoustic_baseline"
    )

    print("\n=======================================================")
    print(">>> EXPERIMENT 2: Advanced Feature Engineering (232 features)")
    print("=======================================================")
    run_experiment_pipeline(
        os.path.join(feat_dir, "engineered_features_train.csv"),
        os.path.join(feat_dir, "engineered_features_test.csv"),
        exp_name="v2_engineered_features"
    )
