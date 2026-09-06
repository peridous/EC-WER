# Release notes

This release contains binary edit-change EC-WER and Direct Estimate.
Source paths below are relative to the original research repository.

| Public file | Provenance and public-copy changes |
| --- | --- |
| `scripts/build_contextual_edit_fsc.py` | From `scripts/edit_consequence_v2/build_contextual_edit_fsc.py`; dynamic helper loading replaced with a sibling import. Alignment, corruption, splitting, repair, and binary labels are unchanged. |
| `scripts/fsc_evaluator.py` | Six helper functions and dependencies from `scripts/final/build_full_frame_hypaware_data.py`: normalization, FSC teacher construction, prediction, and frame extraction. Function ASTs match the originals. |
| `scripts/prepare_training_data.py` | Formats contextual edits without changing labels or order. Creates one Direct Estimate row per unique reference/hypothesis pair and rebuilds the FSC teacher for reference frames. |
| `scripts/train_edit_conditioned_classifier.py` | From `scripts/experiments/train_edit_conditioned_classifier.py`; default seed changed from 20260802 to 20260807 to match the saved EC run. Training behavior is otherwise unchanged. |
| `scripts/train_direct_failure_pair.py` | From `scripts/experiments/train_direct_failure_pair.py`; input reading explicitly uses UTF-8. |
| `scripts/evaluate_ecwer.py` | Adapted from `scripts/experiments/evaluate_edit_conditioned_wer_correct_tokenizer.py`, using the released alignment and formatter. Computes continuous EC-WER, WER, and optional Direct Estimate. Accepts paths via CLI and writes per-edit probabilities and optional pooled metrics. |
| `.gitignore` | Excludes data files, audio, models, checkpoints, results, credentials, environments, caches, and logs. |
| `requirements.txt` | Direct third-party imports plus Trainer and DeBERTa tokenizer runtime dependencies. |
| `data/README.md` | Dataset non-redistribution notice, input schema, and transfer evaluation scope. |
| `README.md` | Method definition, runnable core reproduction commands, scoring interface, and limitations. |
| `RELEASE_NOTES.md` | Script provenance and validation. |

## Verification

- All 28,864 training and 3,626 validation edit texts and labels match the
  original formatted artifacts exactly, in order. Every label equals the
  stored hypothesis/repaired-hypothesis frame inequality.
- Original Direct Estimate artifacts contain 15,840 training and 2,000
  validation pairs. Their pair sets match the unique contextual-data pairs;
  their hypothesis frames match the contextual teacher frames, and every
  failure label equals reference/hypothesis frame inequality.
- SHA-256 hashes of the three copied scripts and source helper remained
  unchanged after copying and verification. The original files were read only.
- All six extracted helper functions match their original ASTs.
- Exhaustive small-input checks passed for 368 individual edit repairs.
- Scoring integration with fixed probabilities passed for substitutions,
  deletions, insertions, clean transcripts, and empty hypotheses. It verifies
  continuous probability aggregation and Direct Estimate output, without
  loading model weights.
- `python -m compileall scripts` passed on Python 3.10.9.
- Working-tree text scans found no password/API-key assignments, recognizable
  Hugging Face or other credential patterns, or research-machine absolute paths.
  No files larger than 10 MB were found, including ignored files and Git metadata.
- Git's index is unchanged. The proposed release contains one modified README
  and ten untracked files; compilation caches are ignored.

Full training and real-checkpoint inference have not been run in the public
environment. `datasets`, `accelerate`, and `sentencepiece` were absent during
verification. The release does not claim to reproduce all paper tables or
recover exact package versions from the research environment.
