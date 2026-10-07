# SHL Spoken Grammar Scoring Engine

Predict a 0–5 grammar score (MOS Likert, half-point steps) for 45–60 s English speech clips.
769 training clips, 216 test clips. The leaderboard reports **RMSE (lower is better)**.

**Final model (v8): CV RMSE 0.485 · training RMSE 0.156 · public leaderboard RMSE 0.359** (CV on non-zero clips; see Iterations)

Main notebook: [`notebooks/SHL_Grammar_Scoring.ipynb`](notebooks/SHL_Grammar_Scoring.ipynb). It runs end-to-end on a Kaggle GPU (T4) with internet enabled. Sections v5 → v8 build on each other.

## Approach

The label measures *grammar*, which lives in **what** a speaker says and in how proficient their speech is overall. Hand-crafted acoustic features (MFCCs, pitch, spectral shape) only capture this indirectly, so the final model combines three information sources:

1. **Audio embeddings.** Frozen pretrained speech encoders: `microsoft/wavlm-base-plus`, `microsoft/wavlm-large`, and the encoders of Whisper `medium` and `large-v3`. For each layer, hidden states are pooled over time (mean + std). The best 3 layers per encoder are chosen by cross-validation (WavLM-large 19–21, Whisper-medium 21/23/24, Whisper-large-v3 29–31). Deep layers work best, which fits a label about language structure rather than sound.
2. **Transcript.** Whisper `large-v3` (via `faster-whisper`) with a deliberately disfluent `initial_prompt`, so Whisper keeps grammar errors and fillers instead of silently correcting them. Word timestamps give fluency statistics: speech rate, pauses, fillers, repetitions, and ASR word confidence.
3. **Grammar model.** `textattack/roberta-base-CoLA` (RoBERTa fine-tuned on the Corpus of Linguistic Acceptability) scores every transcribed sentence as grammatical or not. These scores are aggregated per clip (mean, min, fraction unacceptable), and its pooled hidden state is used as a text embedding.

Each feature block gets a small, regularised model: RidgeCV per layer, an RBF-SVR on each encoder's averaged best layers, and RidgeCV / SVR / shallow LightGBM on the text and fluency features. A **non-negative linear regression** stacks their out-of-fold predictions.

**Training data.** The 37 clips labelled 0 (not on the rubric; transcripts show off-task or non-spontaneous speech, with low ASR confidence) are excluded from training in v8. Error analysis showed the model never predicted a test clip below about 1.7, which suggests the test set contains no such clips. Removing them lowered both CV and leaderboard RMSE.

## Evaluation

* 5-fold stratified CV (strata = binned labels), repeated 3× with different seeds. All scaling and fitting happens inside folds.
* The stacker is scored with its own cross-validation over the OOF matrix, so the reported CV is not optimistic.
* **Training RMSE** (refit on all data, scored on the same data) is reported alongside OOF RMSE, as required.

Base-model results (v7, all 769 clips):

| Model (block) | OOF Pearson | OOF RMSE |
|---|---|---|
| WavLM-large best layers + RBF-SVR | 0.904 | 0.531 |
| WavLM-large layer 21 + Ridge | 0.902 | 0.536 |
| Whisper-medium encoder layer 24 + Ridge | 0.899 | 0.544 |
| Whisper-large-v3 encoder best layers + RBF-SVR | 0.894 | 0.558 |
| WavLM-base-plus (all layers) + Ridge | 0.888 | 0.569 |
| Grammar/fluency scalars + LightGBM | 0.768 | 0.794 |
| CoLA text embedding + Ridge | 0.677 | 0.912 |
| **Stack (v7)** | **0.917** | **0.496** |

Final model (v8, trained on the 732 non-zero clips): **CV RMSE 0.485** (v7 on the same clips: 0.494), CV Pearson 0.878, **training RMSE 0.156**. Pearson is lower than for v7 only because the 0-labelled clips, which stretch the score range, are no longer in the evaluation set.

## Iterations

| Version | Idea | Public LB RMSE |
|---|---|---|
| v1 | 211 hand-crafted acoustic features (MFCC, spectral, pitch, energy) + LightGBM | 0.7528 |
| v1 ensemble | LightGBM + XGBoost + CatBoost + Ridge, SLSQP-weighted | 0.7450 |
| v2 | Engineered acoustic interaction features + LightGBM | 0.7378 |
| v2 + text | Shallow text statistics (TTR, fillers, readability) from `whisper-tiny.en` | 0.6361 |
| v5 | WavLM-base embedding + Whisper large-v3 transcripts + CoLA grammar model, stacked | 0.4038 |
| v6 | + WavLM-large and Whisper-medium encoder, per-layer embeddings, best layers by CV | 0.3740 |
| v7 | + Whisper-large-v3 encoder + RBF-SVR on each encoder's best layers | 0.3710 |
| **v8** | **v7 retrained without the 37 off-task label-0 clips** | **0.359** |

**Lessons learned**

* Acoustic features plateaued around 0.74–0.75 RMSE. Adding more of them did not help.
* Text helped even when the transcripts came from a weak ASR model, which pointed toward the transcript as the missing signal.
* The early ensembles tuned their weights on the same OOF predictions they were scored on. That makes CV look better than it is. v5 evaluates the stacker with its own CV.
* A pretrained speech encoder (WavLM) captures proficiency far better than hand-crafted spectral features. Larger encoders and choosing the best layer, instead of averaging all layers, helped further.
* Error analysis by label showed the remaining error is spread across all score levels (low scores over-predicted, high scores under-predicted). That pointed to representation quality, not a single bad group. It also revealed that the label-0 clips have no counterpart in the test set.

## Repository layout

```
notebooks/
  SHL_Grammar_Scoring.ipynb        # final solution + report, v5 → v8 (run on Kaggle GPU)
  archive/v1_v2_acoustic.ipynb     # earlier acoustic-only pipeline (v1/v2)
src/                               # v1/v2 acoustic feature + model code (kept for reference)
submissions/                       # one CSV per leaderboard submission
figures/                           # plots from earlier versions
```

## Reproduce

1. Create a Kaggle notebook attached to the competition data, with **GPU** and **Internet** enabled.
2. Import `notebooks/SHL_Grammar_Scoring.ipynb` and **Run All**. Data paths are auto-detected.
3. Runtime is about 3 h on a T4: Whisper large-v3 transcription plus four speech encoders. Intermediate results are cached in `./cache`.

## Limitations / next steps

* ASR errors leak into the grammar features. Text is judged from the transcript, not directly from the audio.
* The CoLA model was trained on written sentences, not spontaneous speech.
* When a clip contains little clear speech, Whisper sometimes echoes its own `initial_prompt` text into the transcript (visible in a few label-0 clips). This affects only a handful of clips but is a known side-effect of prompting.
* Possible next steps: fine-tuning WavLM-large end-to-end on the audio (instead of frozen embeddings), and fine-tuning a text encoder (DeBERTa-v3) on the transcripts.
