# Consequence Distillation for Transferable Automatic Speech Recognition Evaluation

Official implementation of **“Consequence Distillation for Transferable Automatic Speech Recognition Evaluation.”**

**Michael Wang** and **Seong Jong Yoo**

## Overview

The release includes two models:

* **Direct Estimate:** transcript-level prediction of downstream task failure.
* **EC-WER:** edit-level consequence prediction for WER-aligned substitutions, deletions, and insertions.

## Binary edit-change EC-WER

Compute an ordinary minimum-WER alignment between reference R and hypothesis
H. For each substitution, deletion, or insertion e, repair only that edit
to obtain H^(-e). The frozen downstream evaluator g supplies the label:

```text
q_chg(e; R, H) = 1[g(H^(-e)) != g(H)]
q_hat_e = P(q_chg = 1 | R, H, e)
EC-WER = sum_e(q_hat_e) / N
```

N is the number of normalized reference words. A DeBERTa-v3-small sequence
classifier predicts continuous positive-class probabilities. Matches do
not contribute. Scoring uses deterministic unit-cost alignment, with ties
resolved by diagonal, deletion, then insertion. Empty normalized references
are rejected; empty hypotheses are supported.

Direct Estimate uses the same reference/hypothesis pairs and predicts
transcript-level downstream failure with native sentence-pair encoding.
Its label compares the teacher frame for H with the teacher frame for R.
The FSC builder retains only commands whose reference prediction matches
the gold frame, so this comparison represents teacher failure on H.

## Reproduction

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -r requirements.txt
python scripts/build_contextual_edit_fsc.py --fsc_train_csv /path/to/train_data.csv --output_dir data/contextual_edits --seed 20260807
python scripts/prepare_training_data.py --input_dir data/contextual_edits --fsc_train_csv /path/to/train_data.csv --edit_output_dir data/edit_training --direct_output_dir data/direct_training --seed 20260807
python scripts/train_edit_conditioned_classifier.py --train_jsonl data/edit_training/train.jsonl --valid_jsonl data/edit_training/valid.jsonl --output_dir models/ecwer --seed 20260807
python scripts/train_direct_failure_pair.py --train_jsonl data/direct_training/train.jsonl --valid_jsonl data/direct_training/valid.jsonl --output_dir models/direct --seed 20260807
python scripts/evaluate_ecwer.py --input /path/to/evaluation.csv --model_dir models/ecwer/best_model --direct_model_dir models/direct/best_model --output_dir results/evaluation
```

Keep the same FSC CSV and seed for both preparation steps. Defaults match
the inspected EC run: 80 training and 40 validation corruptions per command,
20% command-level validation split, four epochs, batch size 16, learning
rate 2e-5, and maximum input length 128. Both students use inverse-frequency
class-weighted cross-entropy and select the checkpoint by validation AUPRC.

The scorer writes `scored.csv`, per-edit probabilities in `edit_scores.csv`,
and `metrics.json`. Input columns default to `reference` and `hypothesis`;
column names are configurable. Optional binary `downstream_failure` labels
produce pooled AUROC/AUPRC for all transcripts and erroneous transcripts.
Metrics are null when a subset lacks both classes. Omit `--direct_model_dir`
to score EC-WER alone. The tokenizer defaults to the original DeBERTa base
tokenizer, matching the inspected corrected evaluation pipeline; specify
`--tokenizer_name` and `--direct_tokenizer_name` if training with another one.

FSC, SLURP, and SLUE-VoxPopuli are not redistributed. See
[data/README.md](data/README.md) for input requirements and dataset scope.
Models, generated datasets, ASR outputs, and evaluation results are excluded
from the release. Reproducing the paper tables requires the original ASR outputs,
transfer-task evaluators, and experiment environment.

## Release provenance and verification

The released scripts were validated through compilation, artifact comparisons, and fixed-probability scoring checks; full end-to-end retraining and checkpoint inference were not rerun for this release.

[RELEASE_NOTES.md](RELEASE_NOTES.md) documents script provenance and validation.
Dependency ranges are inferred from the code; the original environment
lockfile is not included.

## Paper

Paper link coming soon.
