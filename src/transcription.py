import os
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_transcription_pipeline(model_name="openai/whisper-tiny.en", device="cpu"):
    """
    Initializes a Whisper ASR pipeline with automatic chunking for long-form speech.
    """
    print(f"Loading Whisper model: {model_name} on {device}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30,
        return_timestamps=False
    )
    return pipe

def transcribe_dataset(df, audio_dir, output_csv, model_name="openai/whisper-tiny.en", device="cpu"):
    """
    Transcribes all audio files in the dataset and saves the transcripts with metadata.
    """
    # If output file already exists and is complete, load it
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        if len(existing_df) == len(df):
            print(f"Transcripts already exist at {output_csv} ({len(existing_df)} records). Loading cached.")
            return existing_df

    pipe = load_transcription_pipeline(model_name=model_name, device=device)
    
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Transcribing ({os.path.basename(audio_dir)})"):
        fn = row['filename']
        wav_path = os.path.join(audio_dir, fn)
        try:
            audio_data, sr = sf.read(wav_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Run transcription
            res = pipe({"array": audio_data.astype(np.float32), "sampling_rate": sr})
            transcript_text = res.get("text", "").strip()
        except Exception as e:
            print(f"Error transcribing {fn}: {e}")
            transcript_text = ""

        rec = {
            'filename': fn,
            'transcript': transcript_text,
            'char_length': len(transcript_text),
            'word_count': len(transcript_text.split())
        }
        if 'label' in row:
            rec['label'] = row['label']
        records.append(rec)

    res_df = pd.DataFrame(records)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved {len(res_df)} transcripts to {output_csv}")
    return res_df


if __name__ == "__main__":
    data_dir = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\Dataset_Final"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    out_train_transcripts = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\transcripts_train.csv"
    out_test_transcripts = r"C:\Users\Somya\.gemini\antigravity\scratch\shl_grammar_scoring\data\features\transcripts_test.csv"
    
    print("Transcribing Train Audio...")
    transcribe_dataset(train_df, os.path.join(data_dir, "train"), out_train_transcripts)
    
    print("Transcribing Test Audio...")
    transcribe_dataset(test_df, os.path.join(data_dir, "test"), out_test_transcripts)
