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
to score EC-WER alone. Empty hypotheses remain empty strings.

EC-WER preserves the training DeBERTa Metaspace tokenizer with
`fix_mistral_regex=False`. It loads tokenizer files from the checkpoint,
then `--tokenizer_path` if the checkpoint lacks them, then the base model
named by `--tokenizer_name`. The same behavior applies to every dataset.
Direct Estimate retains its separate `--direct_tokenizer_name` setting.

The tested tokenizer environment is `transformers==4.57.6` and
`tokenizers==0.22.2`, pinned in `requirements.txt`. EC-WER rejects other
versions and checks a token-ID fingerprint for three synthetic edit inputs
before inference. To run that check without loading model weights:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); from evaluate_ecwer import load_ecwer_tokenizer; load_ecwer_tokenizer('models/ecwer/best_model'); print('Token-ID check passed')"
ECWER_TOKENIZER_PATH=models/ecwer/best_model python -m unittest discover -s tests
```

Before changing these version pins, verify the token-ID check and all
3,626 validation encodings against the training tokenizer.

FSC, SLURP, and SLUE-VoxPopuli are not redistributed. See
[data/README.md](data/README.md) for input requirements and dataset scope.
Models, generated datasets, ASR outputs, and evaluation results are excluded
from the release. Reproducing the paper tables requires the original ASR outputs,
transfer-task evaluators, and experiment environment.

## Release provenance and verification

Verification includes exact supervision and training-data comparisons,
full retraining of both students, and checkpoint inference in an isolated
Python 3.12.3 environment. The EC-WER scorer matches all 3,626 original
validation token-ID sequences. Retraining comparisons allow numerical
variation; they do not establish bit-for-bit model equality.

[RELEASE_NOTES.md](RELEASE_NOTES.md) documents script provenance and validation.
Tokenizer-library versions are pinned and checked at runtime. Other
dependencies are not a complete environment lockfile.

## Paper

Paper link coming soon.
