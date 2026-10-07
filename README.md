\# SHL Spoken Grammar Scoring Engine



Predict a 0–5 grammar score (MOS Likert, half-point steps) for 45–60 s English speech clips.

769 training clips, 216 test clips. The leaderboard reports \*\*RMSE (lower is better)\*\*.



\*\*Final model (v5): CV RMSE 0.523 · CV Pearson 0.907 · training RMSE 0.342 · public leaderboard RMSE 0.4038\*\*



Main notebook: \[`notebooks/SHL\_Grammar\_Scoring\_v5.ipynb`](notebooks/SHL\_Grammar\_Scoring\_v5.ipynb). It runs end-to-end on a Kaggle GPU (T4) with internet enabled.



\## Approach



The label measures \*grammar\*, which lives in \*\*what\*\* a speaker says and in how proficient their speech is overall. Hand-crafted acoustic features (MFCCs, pitch, spectral shape) only capture this indirectly, so v5 combines three information sources:



1\. \*\*Audio embedding.\*\* `microsoft/wavlm-base-plus`, a self-supervised speech model, mean-pooled over time and layers into one 768-d vector per clip. This is the strongest single signal.

2\. \*\*Transcript.\*\* Whisper `large-v3` (via `faster-whisper`) with a deliberately disfluent `initial\_prompt`, so Whisper keeps grammar errors and fillers instead of silently correcting them. Word timestamps give fluency statistics: speech rate, pauses, fillers, repetitions, and ASR word confidence.

3\. \*\*Grammar model.\*\* `textattack/roberta-base-CoLA` (RoBERTa fine-tuned on the Corpus of Linguistic Acceptability) scores every transcribed sentence as grammatical or not. These scores are aggregated per clip (mean, min, fraction unacceptable), and its pooled hidden state is used as a text embedding.



Each feature block gets a small, regularised model (RidgeCV / SVR / shallow LightGBM). A \*\*non-negative linear regression\*\* stacks their out-of-fold predictions.



\## Evaluation



\* 5-fold stratified CV (strata = binned labels), repeated 3× with different seeds. All scaling and fitting happens inside folds.

\* The stacker is scored with its own cross-validation over the OOF matrix, so the reported CV is not optimistic.

\* \*\*Training RMSE\*\* (refit on all data, scored on the same data) is reported alongside OOF RMSE, as required.



| Model (block) | OOF Pearson | OOF RMSE |

|---|---|---|

| WavLM embedding + Ridge | 0.888 | 0.569 |

| Grammar/fluency scalars + LightGBM | 0.768 | 0.794 |

| Grammar/fluency scalars + Ridge | 0.752 | 0.817 |

| CoLA text embedding + Ridge | 0.677 | 0.912 |

| CoLA text embedding + SVR | 0.673 | 0.918 |

| \*\*Stack (v5)\*\* | \*\*0.907\*\* | \*\*0.523\*\* |



Blend weights: WavLM 0.77 · text Ridge 0.19 · scalar Ridge 0.12 · text SVR 0.08 · scalar LightGBM 0.04.



\## Iterations



| Version | Idea | Public LB RMSE |

|---|---|---|

| v1 | 211 hand-crafted acoustic features (MFCC, spectral, pitch, energy) + LightGBM | 0.7528 |

| v1 ensemble | LightGBM + XGBoost + CatBoost + Ridge, SLSQP-weighted | 0.7450 |

| v2 | Engineered acoustic interaction features + LightGBM | 0.7378 |

| v2 + text | Shallow text statistics (TTR, fillers, readability) from `whisper-tiny.en` | 0.6361 |

| \*\*v5\*\* | \*\*WavLM embedding + Whisper large-v3 transcripts + CoLA grammar model, stacked\*\* | \*\*0.4038\*\* |



\*\*Lessons learned\*\*



\* Acoustic features plateaued around 0.74–0.75 RMSE. Adding more of them did not help.

\* Text helped even when the transcripts came from a weak ASR model, which pointed toward the transcript as the missing signal.

\* The early ensembles tuned their weights on the same OOF predictions they were scored on. That makes CV look better than it is. v5 evaluates the stacker with its own CV.

\* A pretrained speech encoder (WavLM) captures proficiency far better than hand-crafted spectral features.



\## Repository layout



```

notebooks/

&#x20; SHL\_Grammar\_Scoring\_v5.ipynb     # final solution + report (run on Kaggle GPU)

&#x20; archive/v1\_v2\_acoustic.ipynb     # earlier acoustic-only pipeline (v1/v2)

src/                               # v1/v2 acoustic feature + model code (kept for reference)

submissions/                       # one CSV per leaderboard submission

figures/                           # plots from earlier versions

```



\## Reproduce



1\. Create a Kaggle notebook attached to the competition data, with \*\*GPU\*\* and \*\*Internet\*\* enabled.

2\. Import `notebooks/SHL\_Grammar\_Scoring\_v5.ipynb` and \*\*Run All\*\*. Data paths are auto-detected.

3\. Runtime is about 1.5 h on a T4; most of it is Whisper large-v3 transcription. Intermediate results are cached in `./cache`.



\## Limitations / next steps



\* ASR errors leak into the grammar features. Text is judged from the transcript, not directly from the audio.

\* The CoLA model was trained on written sentences, not spontaneous speech.

\* Possible next steps: larger speech encoders (WavLM-large, Whisper encoder) with per-layer selection, and fine-tuning a text encoder (DeBERTa-v3) on the transcripts.

