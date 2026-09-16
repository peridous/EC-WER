import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TOKENIZER_NAME = "microsoft/deberta-v3-small"


def normalize_words(text):
    token_re = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
    normalized = str(text).lower().replace("’", "'")
    return token_re.findall(normalized)


def ordinary_alignment(ref, hyp):
    n, m = len(ref), len(hyp)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i, 0] = i

    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j - 1] + 1,
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                )

    ops = []
    i, j = n, m

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref[i - 1] == hyp[j - 1]
            and dp[i, j] == dp[i - 1, j - 1]
        ):
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and j > 0
            and dp[i, j] == dp[i - 1, j - 1] + 1
        ):
            ops.append(("substitution", i - 1, j - 1))
            i -= 1
            j -= 1
            continue

        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(("deletion", i - 1, None))
            i -= 1
            continue

        if j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            ops.append(("insertion", None, j - 1))
            j -= 1
            continue

        raise RuntimeError("Alignment backtrace failed.")

    return list(reversed(ops))


def repair_one_edit(ref_tokens, hyp_tokens, alignment, focal_idx):
    repaired = []

    for idx, (op, ref_idx, hyp_idx) in enumerate(alignment):
        focal = idx == focal_idx

        if op == "match":
            repaired.append(hyp_tokens[hyp_idx])

        elif op == "substitution":
            if focal:
                repaired.append(ref_tokens[ref_idx])
            else:
                repaired.append(hyp_tokens[hyp_idx])

        elif op == "deletion":
            if focal:
                repaired.append(ref_tokens[ref_idx])

        elif op == "insertion":
            if not focal:
                repaired.append(hyp_tokens[hyp_idx])

    return repaired


def build_ec_input(ref_tokens, hyp_tokens, op, ref_token, hyp_token):
    return (
        f"[REFERENCE] {' '.join(ref_tokens)} "
        f"[HYPOTHESIS] {' '.join(hyp_tokens)} "
        f"[EDIT_TYPE] {op} "
        f"[REFERENCE_EDIT] {ref_token} "
        f"[HYPOTHESIS_EDIT] {hyp_token}"
    )


def build_edit_dataframe(df):
    rows = []

    for source_row, row in df.iterrows():
        ref_tokens = normalize_words(row["reference"])
        hyp_tokens = normalize_words(row["hypothesis"])

        alignment = ordinary_alignment(ref_tokens, hyp_tokens)

        for focal_idx, (op, ref_idx, hyp_idx) in enumerate(alignment):
            if op == "match":
                continue

            repaired = repair_one_edit(
                ref_tokens,
                hyp_tokens,
                alignment,
                focal_idx,
            )

            ref_token = (
                ref_tokens[ref_idx]
                if ref_idx is not None
                else "<eps>"
            )

            hyp_token = (
                hyp_tokens[hyp_idx]
                if hyp_idx is not None
                else "<eps>"
            )

            ec_input = build_ec_input(
                ref_tokens,
                hyp_tokens,
                op,
                ref_token,
                hyp_token,
            )

            rows.append(
                {
                    "source_row": source_row,
                    "slurp_id": row.get("slurp_id"),
                    "recording_id": row.get("recording_id"),
                    "model": row.get("model"),
                    "reference": " ".join(ref_tokens),
                    "original_hypothesis": " ".join(hyp_tokens),
                    "repaired_hypothesis": " ".join(repaired),
                    "operation": op,
                    "ref_idx": ref_idx,
                    "hyp_idx": hyp_idx,
                    "ref_token": ref_token,
                    "hyp_token": hyp_token,
                    "ec_input": ec_input,
                    "source_hypothesis_predicted_full_frame":
                        row.get("hypothesis_predicted_full_frame"),
                    "gold_full_frame": row.get("gold_full_frame"),
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def score_ec_model(
    edit_df,
    model_dir,
    batch_size=64,
    max_length=128,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    ).to(device)

    model.eval()

    probs = []
    texts = edit_df["ec_input"].tolist()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        logits = model(**encoded).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, 1]

        probs.extend(batch_probs.cpu().numpy().tolist())

    edit_df = edit_df.copy()
    edit_df["ec_probability"] = probs

    return edit_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "results/contextual_edit_fsc_v1_correcttok_lambda32/"
            "slurp/scored.csv"
        ),
    )

    parser.add_argument(
        "--model_dir",
        default="results/contextual_edit_fsc_v1/best_model",
    )

    parser.add_argument(
        "--output_dir",
        default="results/slurp_edit_transfer_full",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading:", args.input)
    df = pd.read_csv(args.input)

    eligible = df[
        (df["full_frame_reference_correct"] == True)
        & (df["wer"] > 0)
    ].copy()

    print("Eligible erroneous hypotheses:", len(eligible))

    edit_df = build_edit_dataframe(eligible)

    print("Total real ASR edits:", len(edit_df))
    print(edit_df["operation"].value_counts())

    edit_df = score_ec_model(
        edit_df,
        args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(
        "EC qhat range:",
        edit_df["ec_probability"].min(),
        "to",
        edit_df["ec_probability"].max(),
    )

    output_path = Path(args.output_dir) / (
        "slurp_edit_pairs_with_ec.csv"
    )

    edit_df.to_csv(output_path, index=False)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
