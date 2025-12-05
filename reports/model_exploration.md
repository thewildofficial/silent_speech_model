# Silent Speech Model – Exploration Report

## TL;DR
- Goal: document what we have (data, models, LMs, vocoder), what we ran, and why results currently look poor (collapse to “aa”).
- Status: LM built, parallel voiced data present, HiFi-GAN imports fixed, eval runs. Mismatch between checkpoint outputs (80 classes) and the 38-class grapheme vocab used in eval likely drives the collapse.
- Next: use a matching 38-class checkpoint or align vocab/LM to 80 classes, then run proper decoding + WER.

## Background for new teammates
- Task: map EMG (silent speech) to text. The model predicts acoustic features (mel-like), then alignment/decoding turns those into text.
- Data: parallel voiced audio is available for the silent examples, which lets us align predictions to “true” audio frames.
- Metrics: current script reports DTW-based phone confusion (alignment sanity check). WER requires decoding to text (not enabled yet in this eval path).

## Artifacts and environments
- Repo: `/home/reinforce/Documents/Project Elysium/silent_speech_model`
- Virtualenv: `.venv`
- Processed data: `/media/reinforce/B4FE40D4FE409092/xinference_artifacts/processed_data` (dev/test ready)
- Normalizers: `normalizers.pkl`
- HiFi-GAN: `/media/.../models/pretrained_models/hifigan_finetuned/{checkpoint,config.json}`
- Transduction checkpoints:
  - `/media/.../transduction_model.pt` → `w_out` `[80, 768]` (80 classes; likely mismatched to current vocab)
  - `artifacts/model.pt` → `w_out` `[38, 768]` (matches 38 grapheme vocab + blank)

## Model architecture (what we are evaluating)
- Base model: Transformer encoder over EMG features (see `architecture.py`, `Model` class).
- Input: EMG frames (and raw EMG for conv stack); output: log-softmax over classes via `w_out`.
- Decoding options:
  - DTW path (used here): Align predicted acoustic features to reference voiced features; derive phone confusion.
  - CTC + LM path (available in `architecture.py`): Beam decode logits with lexicon + KenLM.
- Aux pieces: S4 variant exists; HiFi-GAN vocoder can turn predicted mels into audio for downstream ASR.

## Vocabulary and text transform
- Grapheme mode: `TextTransform(togglePhones=False)` → 36 graphemes (a–z, 0–9) + separator `|`; CTC blank adds 1 → expected 38 outputs.
- Padding hack removed: we no longer pad `text_transform.chars`; we warn if checkpoint outputs ≠ vocab+blank.
- Phones mode exists (togglePhones=True) but is not used in this run.

## Language model status
- Built KenLM (`kpu/kenlm`) at `/media/.../lmbuild_binary/kenlm/build/bin`.
- Generated `lm.binary` from `librispeech-4gram.arpa.gz` (≈3.0 GB) in `/media/.../lm`.
- Lexicon files present: `librispeech-lexicon.txt`, `cmudict.txt`.

## Data integrity – parallel voiced audio
- Checked `.mat` files in dev/test: all have `parallel_voiced_audio_features`.
  - Dev: 200/200 present.
  - Test: 99/99 present.
  - Example (test/session_0/1.mat): keys include `audio_features`, `emg`, `text`, `text_int`, `phonemes`, `parallel_voiced_audio_features (255×80)`, `parallel_voiced_raw_emg`.

## HiFi-GAN status
- Cloned upstream HiFi-GAN into `hifi_gan/` and added `__init__.py`.
- Imports validated (`env.AttrDict`, `models.Generator` load). Checkpoint/config live in `hifigan_finetuned/`.

## Evaluation we ran (test split)
- Command:
  ```
  source .venv/bin/activate
  SCRATCH=/media/reinforce/B4FE40D4FE409092/xinference_artifacts \
  PROJECT_FOLDER="/home/reinforce/Documents/Project Elysium/silent_speech_model" \
  NORMALIZERS_FILE=normalizers.pkl \
  python evaluate.py \
    --base_dir /media/reinforce/B4FE40D4FE409092/xinference_artifacts/processed_data \
    --models /media/reinforce/B4FE40D4FE409092/xinference_artifacts/models/pretrained_models/transduction_model.pt \
    --normalizers_file normalizers.pkl \
    --output_directory output_run1 \
    --lm_directory /media/reinforce/B4FE40D4FE409092/xinference_artifacts/lm \
    --hifigan_checkpoint /media/reinforce/B4FE40D4FE409092/xinference_artifacts/models/pretrained_models/hifigan_finetuned/checkpoint \
    --dev=False
  ```
- Device: CUDA if available; otherwise CPU.
- Warning: checkpoint outputs 80 vs expected 38 (vocab+blank). Proceeded anyway.
- Output (DTW phone confusion; no audio/Whisper decoding in this path):
  - Collapse toward `aa`; common confusions:
    - d↔aa 73.4 / accuracy 26.6
    - l↔aa 73.6 / 26.4
    - er↔aa 73.9 / 26.1
    - ih↔aa 76.3 / 23.7
    - iy↔aa 77.6 / 22.4
    - n↔aa 81.1 / 18.9
    - s↔aa 81.3 / 18.7
    - t↔aa 82.6 / 17.4
    - ah↔aa 83.9 / 16.1
    - sil↔aa 97.3 / 2.7
- Log: `output_run1/eval_log.txt` (captures the output-dimension warning).

## Attempt to swap to the 38-class checkpoint
- Goal: rerun with `artifacts/model.pt` (38 outputs) to avoid the 80-vs-38 mismatch.
- Result: run failed in DTW with `RuntimeError: X1 and X2 must have the same number of columns. X1: 38 X2: 80`.
  - Cause: this checkpoint produces 38 text logits, but the DTW path expects 80-dim acoustic features to compare against the 80-dim reference mel features. The 80-class checkpoint was evidently trained to predict mel features directly; the 38-class checkpoint is a text-output model, so its outputs cannot be DTW-aligned to mel features.
- Implication: To evaluate the 38-class checkpoint we need a text-level decoding path (CTC + LM) instead of DTW-to-mels, or a mel-predictor checkpoint with the matching output dimension.

## Interpretation
- Likely root cause: output-label mismatch (80-class checkpoint vs 38-class eval vocab). Even with correct parallel audio and LM present, the model collapses to a single class in alignment.
- LM is built but not used inside the DTW confusion path; LM only helps decoding (CTC/beam or ASR on synthesized audio).
- HiFi-GAN is ready; current eval script skips synthesis and Whisper, so we do not yet have WER.

## How to get Word Error Rate (WER)
- Use a checkpoint that matches the vocab (prefer the 38-output `artifacts/model.pt`).
- Option A: CTC + LM decode directly from logits (architecture has a beam decoder with `lm.binary` and lexicon).
- Option B: Generate audio → transcribe with Whisper:
  1) Run model to produce mels; invert normalization.
  2) Vocoder with HiFi-GAN to WAV.
  3) Transcribe with Whisper; compute WER with `jiwer.wer(ref, hyp)` after consistent text normalization.
- Report overall and per-split WER; log a few example hypotheses vs references.

## Recommendations / next steps
- Swap to the 38-class checkpoint (`artifacts/model.pt`) and re-run the eval to see if the collapse disappears.
- If sticking with the 80-class checkpoint, align the vocab: update `TextTransform`/lexicon/LM to 80 labels and regenerate `lm.binary`.
- Add decoding + WER reporting (either CTC+LM or audio→Whisper) so we have end-to-end metrics.
- Persist full confusion matrices and token error rates to disk (JSON/CSV) for dev/test for reproducibility.
- Rerun on GPU if available to remove CPU slowdown and confirm no numerical quirks.

## Synthetic EMG datasets (external)
- Sources (downloaded to `artifacts/synthetic/`):
  - Noisy EMG simulation: https://drive.google.com/file/d/1a3qsCexT7Nn3aKfmojW3lK9bAEsXhhOE/view?ts=693307a4 → `noisy_emg_dataset.zip`
  - Clean EMG simulation: https://drive.google.com/file/d/1T0uGsmWrQxn-RzhB1omnnoh_YbLLlX2p/view?ts=69330784 → `clean_emg_dataset.zip`
- Contents after extraction:
  - Each zip has a `manifest.csv` with columns `sample_id, filename, env_filename, duration_s, num_channels, fs, num_samples` and 200 samples.
  - Each sample has two files: `<id>.npy` (raw EMG) and `<id>_env.npy` (envelope). Shapes observed:
    - Noisy sample_0000: `(6103, 8)` float32, min -1.48, max 1.63; env `(6103, 8)` float32 in [0,1].
    - Clean sample_0000: `(2979, 8)` float32, min -0.85, max 0.88; env `(2979, 8)` float32 in [~0.003,1].
  - Sampling rate in manifest: 2048 Hz; 8 channels.
- Compatibility with current pipeline:
  - Our datasets expect paired audio/text/phonemes and aligned `.mat` or raw folders with `_info.json` + audio; these synthetic sets provide only EMG + envelope, no audio or transcripts.
  - The DTW path needs reference audio features; the CTC path needs text targets. Neither is available here, so we cannot compute loss, DTW, or WER.
  - Channel count (8) matches model input expectation, but resampling/normalization would be needed (pipeline typically downsamples to ~517 Hz and normalizes).
- Conclusion: The synthetic EMG zips are readable (NumPy arrays) but not directly usable for model training/eval without adding synthetic transcripts and reference audio (or constructing suitable labels) and adapting preprocessing to their 2048 Hz sampling rate.

## FAQ
- What is DTW confusion? A phone-level confusion matrix after aligning predicted acoustic features to reference voiced features with Dynamic Time Warping; it is a sanity check, not WER.
- Why the “aa” collapse? The checkpoint outputs 80 classes but eval expects 38, so labels are misaligned and DTW maps many tokens to one class.
- Does the LM help DTW? No. LM helps decoding; DTW uses acoustic distances plus phone logits, not the LM.
- How do we get WER? Decode to text (CTC+LM or vocoded audio → ASR) and compare hypotheses to references with `jiwer.wer`.
- Are parallel voiced features present? Yes—dev and test splits both have `parallel_voiced_audio_features`, so alignment should work once labels match.