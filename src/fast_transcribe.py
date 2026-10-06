import os
import time
import pandas as pd
import numpy as np
import whisper
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch

def transcribe_batch_worker(model, audio_files, audio_dir):
    results = []
    for fn in audio_files:
        wav_path = os.path.join(audio_dir, fn)
        try:
            # Whisper direct transcribe
            res = model.transcribe(wav_path, fp16=False, language="en")
            txt = res.get("text", "").strip()
        except Exception as e:
            print(f"Error {fn}: {e}")
            txt = ""
        results.append({
            'filename': fn,
            'transcript': txt,
            'char_length': len(txt),
            'word_count': len(txt.split())
        })
    return results

def run_fast_transcription():
    base_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring"
    data_dir = os.path.join(base_dir, "data", "Dataset_Final")
    feat_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(feat_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_out = os.path.join(feat_dir, "transcripts_train.csv")
    test_out = os.path.join(feat_dir, "transcripts_test.csv")

    torch.set_num_threads(4)
    print("Loading official OpenAI Whisper model (tiny.en)...")
    model = whisper.load_model("tiny.en")

    # 1. Train Transcriptions
    print("\n--- Transcribing Training Audio Files ---")
    train_results = []
    processed_train = {}
    if os.path.exists(train_out):
        existing = pd.read_csv(train_out)
        if len(existing) == len(train_df):
            print(f"Train transcripts already complete ({len(existing)} records).")
            train_results = existing.to_dict('records')
        else:
            processed_train = {r['filename']: r for r in existing.to_dict('records')}
    elif os.path.exists(train_out + ".partial"):
        try:
            p_df = pd.read_csv(train_out + ".partial")
            processed_train = {r['filename']: r for r in p_df.to_dict('records')}
            print(f"Loaded {len(processed_train)} from partial checkpoint.")
        except Exception:
            pass

    if len(train_results) != len(train_df):
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train Whisper"):
            fn = row['filename']
            if fn in processed_train:
                rec = processed_train[fn]
                if 'label' in row and 'label' not in rec:
                    rec['label'] = row['label']
                train_results.append(rec)
                continue
            
            wav_path = os.path.join(data_dir, "train", fn)
            try:
                res = model.transcribe(wav_path, fp16=False, language="en")
                txt = res.get("text", "").strip()
            except Exception as e:
                print(f"Error {fn}: {e}")
                txt = ""
            
            rec = {
                'filename': fn,
                'transcript': txt,
                'char_length': len(txt),
                'word_count': len(txt.split()),
                'label': row['label']
            }
            train_results.append(rec)

            if len(train_results) % 25 == 0:
                pd.DataFrame(train_results).to_csv(train_out + ".partial", index=False)

        res_train_df = pd.DataFrame(train_results)
        res_train_df.to_csv(train_out, index=False)
        if os.path.exists(train_out + ".partial"):
            os.remove(train_out + ".partial")
        print(f"Saved {len(res_train_df)} train transcripts to {train_out}")

    # 2. Test Transcriptions
    print("\n--- Transcribing Test Audio Files ---")
    test_results = []
    processed_test = {}
    if os.path.exists(test_out):
        existing = pd.read_csv(test_out)
        if len(existing) == len(test_df):
            print(f"Test transcripts already complete ({len(existing)} records).")
            test_results = existing.to_dict('records')
        else:
            processed_test = {r['filename']: r for r in existing.to_dict('records')}

    if len(test_results) != len(test_df):
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Whisper"):
            fn = row['filename']
            if fn in processed_test:
                test_results.append(processed_test[fn])
                continue
            
            wav_path = os.path.join(data_dir, "test", fn)
            try:
                res = model.transcribe(wav_path, fp16=False, language="en")
                txt = res.get("text", "").strip()
            except Exception as e:
                print(f"Error {fn}: {e}")
                txt = ""
            
            rec = {
                'filename': fn,
                'transcript': txt,
                'char_length': len(txt),
                'word_count': len(txt.split())
            }
            test_results.append(rec)

            if len(test_results) % 25 == 0:
                pd.DataFrame(test_results).to_csv(test_out + ".partial", index=False)

        res_test_df = pd.DataFrame(test_results)
        res_test_df.to_csv(test_out, index=False)
        if os.path.exists(test_out + ".partial"):
            os.remove(test_out + ".partial")
        print(f"Saved {len(res_test_df)} test transcripts to {test_out}")

    print("\n>>> ALL TRANSCRIPTIONS COMPLETE!")

if __name__ == "__main__":
    run_fast_transcription()
