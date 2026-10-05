# Transcribing code-switched Indonesian–Javanese speech

My entry for the Indonesian–Javanese track of the **Lost in Transcription Challenge**: automatic speech recognition (ASR) for spontaneous conversations in which speakers switch between Indonesian and Javanese, sometimes mid-sentence.

> **Status:** work in progress (October 2026). Zero-shot pipeline running end to end; model upgrade and fine-tuning next.

| Run | Model | Dev WER |
|---|---|---|
| Starter baseline | faster-whisper small, language auto-detected, VAD on | 79.2%\* |
| Diagnosis-driven decoding | same model, settings below | **35.4%** |

<sub>\*Scored with my reimplementation of the competition's documented normalizer; the 35.4% run was scored with the official `score.py`. Dev set: 372 clips, ~2.5 hours, 4 speakers, 17,473 reference words. Lower is better.</sub>

## Why this problem is hard

Indonesian and Javanese share much of their vocabulary, so many words can't be cleanly assigned to one language. Off-the-shelf models like Whisper expect each clip to be in one language. They also expect formal, written-style text, while the references here are transcribed spontaneous speech: fillers (*e...*), truncated words, reduplications (*kanca-kanca*), and colloquial spellings (*nggak*, *gitu*). On top of that, the metric is strict. Word error rate is case-sensitive mid-sentence, and any spelling difference counts as a full error.

## What I did

I treated the baseline as an experiment to diagnose before changing anything. I scored it with a normalizer matching the official one, broke errors into substitutions, deletions, and insertions, and read the worst clips side by side with their references. Three failure modes accounted for most of the error:

1. **Wrong-language decoding.** 132 of 372 outputs contained non-Latin scripts (Cyrillic, Chinese, Korean, Hebrew), plus Portuguese, German, and English words. Per-clip language auto-detection was misfiring on code-switched speech.
2. **Missing speech.** Deletions alone were 26.4% of reference words. 75 clips came back with less than half the reference length, and 4 were empty. Voice-activity filtering was cutting quiet speech, and long clips (up to 40 s, beyond Whisper's 30 s window) were losing their tails.
3. **Convention mismatch.** The references contain no digits; the outputs contained 228 digit characters. The model wrote *2019* where transcribers wrote *dua ribu sembilan belas*.

The changes target each one directly:

- Force Indonesian decoding instead of auto-detection.
- Turn off VAD, and stop conditioning on previous text, so one bad 30-second window can't derail the rest of a long clip. Add temperature fallback and compression-ratio checks against repetition loops.
- Add a short initial prompt in the references' colloquial style to nudge spelling.
- Post-process: spell out numbers in Indonesian (including ordinals, decimals, thousands separators, and percentages), drop non-Latin tokens, and collapse runaway repetitions.

Together these brought dev WER from 79.2% to 35.4% without changing the model.

## Repository layout

```
main.py               Inference entrypoint run by the competition container
evaluate_dev.py       WER with official-style normalization, per-language and S/D/I breakdown
prepare_dev_data.py   Lays out the dev set as the runtime expects; builds ground_truth.csv
sync_submission.sh    Copies main.py + model/ into the runtime repo for packaging
model/                CTranslate2 Faster-Whisper weights (not committed)
results/              Predictions from each experiment
```

The official [runtime repository](#) is kept as an untouched sibling clone. This repo only contains my code, and `sync_submission.sh` copies the submission files across at packaging time.

## Running it

```bash
# Download a CTranslate2 model into model/
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir='model')"

# Lay out the dev set inside the runtime repo
python prepare_dev_data.py --dev-dir ~/Downloads/indonesian_dev --runtime-data ../runtime/data/data

# Transcribe (CPU locally; the competition runtime uses a GPU)
LIT_DATA_DIR=../runtime/data/data LIT_SUBMISSION_DIR=results/run1 python main.py

# Score
python evaluate_dev.py --predictions results/run1/submission.csv \
    --reference ../runtime/data/data/metadata.tsv --worst 10
```

Decoding settings can be changed for A/B tests without editing code: `LIT_LANGUAGE` (`id`, `jw`, `auto`), `LIT_PROMPT` (empty string disables), `LIT_BEAM`, `LIT_VAD`, `LIT_POSTPROC`.

## Limitations

- **No ablation yet.** All four changes were applied together, so I can't yet say how much each one contributed. Isolating them is the next experiment.
- **Small dev set.** Four speakers in two conversations. The number is a sanity check, not a precise estimate of test performance.
- **Indonesian forcing is a blunt tool.** It fixed wrong-script output, but it may push Javanese words toward Indonesian spellings. The per-language breakdown will show whether that's happening.

## Next steps

1. Ablate each decoding change and keep only what helps.
2. Swap in large-v3 or large-v3-turbo within the submission size and two-hour runtime limits.
3. Fine-tune on the Jember Javanese Spontaneous Speech Corpus (~10 hours, same dialect family), so the model learns the transcription conventions instead of being prompted toward them.
4. Validate in the official container through the platform's smoke tests.

## Acknowledgements

Challenge, development data, and runtime provided by the Lost in Transcription Challenge organizers. Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI's Whisper models.
