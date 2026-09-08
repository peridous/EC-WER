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

The tested tokenizer environment is `transformers==4.57.6` and
`tokenizers==0.22.2`, pinned in `requirements.txt`. EC-WER rejects other
versions and checks a token-ID fingerprint for three synthetic edit inputs
before inference. To run that check without loading model weights:

```bash
python -c "import sys; sys.path.insert(0, 'scripts'); from evaluate_ecwer import load_ecwer_tokenizer; load_ecwer_tokenizer('models/ecwer/best_model'); print('Token-ID check passed')"
ECWER_TOKENIZER_PATH=models/ecwer/best_model python -m unittest discover -s tests
```

```bash
export ECWER_VALID_JSONL=/path/to/original/valid.jsonl
export ECWER_ORIGINAL_TOKENIZER_JSON=/path/to/original/checkpoint/tokenizer.json
export ECWER_TOKENIZER_PATH=microsoft/deberta-v3-small
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -v
```

FSC, SLURP, and SLUE-VoxPopuli are not redistributed. See
[data/README.md](data/README.md) for input requirements and dataset scope.

## Paper

Paper link coming soon.
