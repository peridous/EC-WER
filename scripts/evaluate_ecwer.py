from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_contextual_edit_fsc import align
from fsc_evaluator import normalize_tokens
from prepare_training_data import build_input_text


def load_ecwer_tokenizer(source):
    for package, expected in (("transformers", "4.57.6"), ("tokenizers", "0.22.2")):
        if version(package) != expected:
            raise RuntimeError(f"EC-WER requires {package}=={expected}; install requirements.txt")
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, fix_mistral_regex=False)
    texts = [
        "[REFERENCE] turn on the light [HYPOTHESIS] turn off the light "
        "[EDIT_TYPE] substitution [REFERENCE_EDIT] on [HYPOTHESIS_EDIT] off",
        "[REFERENCE] hello world [HYPOTHESIS] hello [EDIT_TYPE] deletion "
        "[REFERENCE_EDIT] world [HYPOTHESIS_EDIT] [EMPTY]",
        "[REFERENCE] hello [HYPOTHESIS] hello there [EDIT_TYPE] insertion "
        "[REFERENCE_EDIT] [EMPTY] [HYPOTHESIS_EDIT] there",
    ]
    ids = tokenizer(texts, truncation=True, max_length=128)["input_ids"]
    fingerprint = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != "9487d4a3eb8a9a4933820451f19a44faa66b5c03a8c1d307f099c81ce6e71f5f":
        raise RuntimeError("EC-WER tokenizer does not match the training token IDs")
    return tokenizer


def predict(texts, model_dir, tokenizer_name, positive_label, batch_size, max_length,
            hypotheses=None, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    positive_id = int(model.config.label2id[positive_label])
    result = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            pair = {} if hypotheses is None else {
                "text_pair": hypotheses[start:start + batch_size]
            }
            encoded = tokenizer(
                texts[start:start + batch_size], **pair, padding=True,
                truncation=True, max_length=max_length, return_tensors="pt",
            )
            logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            result.extend(torch.softmax(logits, dim=-1)[:, positive_id].cpu().tolist())
    return np.asarray(result, dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Score transcript CSVs with continuous binary-change EC-WER and Direct Estimate."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--direct_model_dir", type=Path)
    parser.add_argument("--tokenizer_name", default="microsoft/deberta-v3-small")
    parser.add_argument("--tokenizer_path", type=Path,
                        help="EC-WER tokenizer directory when the checkpoint has no tokenizer files.")
    parser.add_argument("--direct_tokenizer_name", default="microsoft/deberta-v3-small")
    parser.add_argument("--reference_col", default="reference")
    parser.add_argument("--hypothesis_col", default="hypothesis")
    parser.add_argument("--failure_col", default="downstream_failure")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("Batch size and maximum length must be positive")
    frame = pd.read_csv(args.input, keep_default_na=False)
    if frame.empty:
        raise ValueError("Input CSV is empty")
    references = frame[args.reference_col].astype(str).tolist()
    hypotheses = frame[args.hypothesis_col].astype(str).tolist()
    texts, owners, records, denominators, counts = [], [], [], [], []
    names = {"S": "substitution", "D": "deletion", "I": "insertion"}
    for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
        ref, hyp = normalize_tokens(reference), normalize_tokens(hypothesis)
        if not ref:
            raise ValueError(f"Empty normalized reference at row {index}")
        edits = [edit for edit in align(ref, hyp) if edit[0] != "M"]
        denominators.append(len(ref))
        counts.append(len(edits))
        for op, ri, hi, pos in edits:
            texts.append(build_input_text(
                " ".join(ref), " ".join(hyp), names[op],
                ref[ri] if ri is not None else "",
                hyp[hi] if hi is not None else "",
            ))
            owners.append(index)
            records.append({"row_position": index, "operation": names[op],
                            "reference_index": ri, "hypothesis_index": hi,
                            "repair_position": pos})
    tokenizer_files = ("tokenizer.json", "spm.model", "sentencepiece.bpe.model", "vocab.json", "vocab.txt")
    tokenizer_source = (
        args.model_dir if any((args.model_dir / name).is_file() for name in tokenizer_files)
        else args.tokenizer_path or args.tokenizer_name
    )
    print(f"EC-WER tokenizer: {tokenizer_source}")
    tokenizer = load_ecwer_tokenizer(tokenizer_source)
    probabilities = predict(texts, args.model_dir, tokenizer_source, "CONSEQUENTIAL",
                            args.batch_size, args.max_length, tokenizer=tokenizer) if texts else np.array([])
    sums = np.zeros(len(frame), dtype=float)
    np.add.at(sums, np.asarray(owners, dtype=int), probabilities)
    frame["wer"] = np.asarray(counts) / np.asarray(denominators)
    frame["ecwer"] = sums / np.asarray(denominators)
    frame["edit_count"] = counts
    for record, probability in zip(records, probabilities):
        record["q_hat"] = float(probability)
    score_columns = ["wer", "ecwer"]
    if args.direct_model_dir:
        frame["direct_estimate"] = predict(
            references, args.direct_model_dir, args.direct_tokenizer_name, "FAILURE",
            args.batch_size, args.max_length, hypotheses=hypotheses,
        )
        score_columns.append("direct_estimate")
    metrics = []
    if args.failure_col in frame:
        labels = pd.to_numeric(frame[args.failure_col], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("Downstream failure labels must be binary")
        for subset, mask in [("all", np.ones(len(frame), dtype=bool)),
                             ("errors_only", frame["wer"].to_numpy() > 0)]:
            y = labels[mask].to_numpy(dtype=int)
            for score in score_columns:
                valid = len(np.unique(y)) == 2
                metrics.append({
                    "subset": subset, "score": score, "rows": len(y),
                    "auprc": float(average_precision_score(y, frame.loc[mask, score])) if valid else None,
                    "auroc": float(roc_auc_score(y, frame.loc[mask, score])) if valid else None,
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "scored.csv", index=False)
    pd.DataFrame(records, columns=["row_position", "operation", "reference_index",
                                  "hypothesis_index", "repair_position", "q_hat"]).to_csv(
        args.output_dir / "edit_scores.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {len(frame)} transcripts and {len(records)} edits: {args.output_dir}")


if __name__ == "__main__":
    main()
