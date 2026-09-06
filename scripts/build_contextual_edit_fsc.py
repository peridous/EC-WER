
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import fsc_evaluator as helper


def align(ref, hyp):
    """Unit-cost WER alignment; ties prefer diagonal, deletion, insertion.

    Return (op, ref_index, hyp_index, hyp_position) tuples for M/S/D/I.
    hyp_position is the insertion index for a deletion repair.
    """
    n, m = len(ref), len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                candidates = [
                    (dp[i - 1][j - 1], 0, "M"),
                    (dp[i - 1][j] + 1, 1, "D"),
                    (dp[i][j - 1] + 1, 2, "I"),
                ]
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, 0, "S"),
                    (dp[i - 1][j] + 1, 1, "D"),
                    (dp[i][j - 1] + 1, 2, "I"),
                ]

            cost, _, op = min(candidates)
            dp[i][j] = cost
            back[i][j] = op

    rev = []
    i, j = n, m

    while i > 0 or j > 0:
        op = back[i][j]

        if op in {"M", "S"}:
            rev.append((op, i - 1, j - 1, j - 1))
            i -= 1
            j -= 1

        elif op == "D":
            rev.append(("D", i - 1, None, j))
            i -= 1

        elif op == "I":
            rev.append(("I", None, j - 1, j - 1))
            j -= 1

        else:
            raise RuntimeError((i, j, op))

    return list(reversed(rev))


def repair_edit(ref, hyp, edit):
    op, ri, hj, hpos = edit
    repaired = list(hyp)

    if op == "S":
        repaired[hj] = ref[ri]

    elif op == "I":
        del repaired[hj]

    elif op == "D":
        repaired.insert(hpos, ref[ri])

    else:
        raise ValueError(op)

    return repaired


def corrupt(tokens, vocab, rng, min_edits=1, max_edits=3):
    out = list(tokens)

    wanted = rng.randint(min_edits, max_edits)

    for _ in range(wanted):
        choices = ["S", "D", "I"]

        if len(out) <= 1:
            choices = ["S", "I"]

        op = rng.choice(choices)

        if op == "S":
            pos = rng.randrange(len(out))
            old = out[pos]

            candidates = [
                word for word in vocab
                if word != old
            ]

            if candidates:
                out[pos] = rng.choice(candidates)

        elif op == "D":
            pos = rng.randrange(len(out))
            del out[pos]

        else:
            pos = rng.randrange(len(out) + 1)
            out.insert(pos, rng.choice(vocab))

    return out


def make_candidates(
    references,
    vocab,
    corruptions_per_reference,
    rng,
):
    hypotheses = []

    for item in references:
        ref_tokens = item["reference"].split()
        seen = set()

        attempts = 0
        target = corruptions_per_reference

        while len(seen) < target and attempts < target * 10:
            attempts += 1

            hyp_tokens = corrupt(
                ref_tokens,
                vocab,
                rng,
            )

            hyp_text = " ".join(hyp_tokens)

            if hyp_text == item["reference"]:
                continue

            if hyp_text in seen:
                continue

            seen.add(hyp_text)

            operations = [
                edit
                for edit in align(ref_tokens, hyp_tokens)
                if edit[0] != "M"
            ]

            if not operations:
                continue

            hypotheses.append({
                **item,
                "hypothesis": hyp_text,
                "hypothesis_tokens": hyp_tokens,
                "alignment": operations,
            })

    return hypotheses


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fsc_train_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train_corruptions",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--valid_corruptions",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260807,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    frame = pd.read_csv(args.fsc_train_csv)

    required = {
        "transcription",
        "action",
        "object",
        "location",
    }

    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Missing FSC columns: {sorted(missing)}"
        )

    commands = {}

    for row in frame.itertuples(index=False):
        reference = helper.normalize_text(
            getattr(row, "transcription")
        )

        if not reference:
            continue

        gold = (
            str(getattr(row, "action")),
            str(getattr(row, "object")),
            str(getattr(row, "location")),
        )

        if reference in commands:
            if commands[reference]["gold_frame"] != gold:
                raise RuntimeError(
                    f"Conflicting frame for {reference}"
                )
        else:
            commands[reference] = {
                "dataset": "fsc",
                "reference": reference,
                "gold_frame": gold,
            }

    references = list(commands.values())

    print("Unique FSC commands:", len(references))

    _, evaluator = helper.build_fsc_resources(
        args.fsc_train_csv,
        args.seed,
    )

    ref_outputs = helper.fsc_predict_texts(
        [x["reference"] for x in references],
        evaluator,
    )

    retained = []

    for item in references:
        output = ref_outputs[item["reference"]]
        pred = helper.fsc_prediction(output)

        if tuple(pred) == tuple(item["gold_frame"]):
            retained.append(item)

    print(
        "Evaluator-A-correct clean commands:",
        len(retained),
        "/",
        len(references),
    )

    rng.shuffle(retained)

    valid_n = max(
        1,
        round(len(retained) * args.valid_fraction),
    )

    valid_refs = retained[:valid_n]
    train_refs = retained[valid_n:]

    print("Student train commands:", len(train_refs))
    print("Student valid commands:", len(valid_refs))

    vocab = sorted({
        token
        for item in train_refs
        for token in item["reference"].split()
    })

    print("Training vocabulary:", len(vocab))

    train_hypotheses = make_candidates(
        train_refs,
        vocab,
        args.train_corruptions,
        rng,
    )

    valid_hypotheses = make_candidates(
        valid_refs,
        vocab,
        args.valid_corruptions,
        rng,
    )

    print(
        "Synthetic hypotheses:",
        "train=",
        len(train_hypotheses),
        "valid=",
        len(valid_hypotheses),
    )

    def label_hypotheses(items, split_name):
        examples = []

        texts = []
        records = []

        for hypothesis_id, item in enumerate(items):
            ref_tokens = item["reference"].split()
            hyp_tokens = item["hypothesis_tokens"]

            h_text = " ".join(hyp_tokens)

            texts.append(h_text)

            for edit_id, edit in enumerate(item["alignment"]):
                repaired_tokens = repair_edit(
                    ref_tokens,
                    hyp_tokens,
                    edit,
                )
                repaired_text = " ".join(repaired_tokens)

                texts.append(repaired_text)

                records.append({
                    "hypothesis_id": hypothesis_id,
                    "edit_id": edit_id,
                    "item": item,
                    "edit": edit,
                    "hypothesis_text": h_text,
                    "repaired_text": repaired_text,
                })

        unique_texts = list(dict.fromkeys(texts))

        print(
            f"{split_name}: evaluator queries "
            f"{len(unique_texts):,}"
        )

        outputs = helper.fsc_predict_texts(
            unique_texts,
            evaluator,
        )

        predictions = {
            text: tuple(
                helper.fsc_prediction(outputs[text])
            )
            for text in unique_texts
        }

        for record in records:
            item = record["item"]
            op, ri, hj, hpos = record["edit"]

            h_frame = predictions[
                record["hypothesis_text"]
            ]
            repaired_frame = predictions[
                record["repaired_text"]
            ]

            label = int(h_frame != repaired_frame)

            ref_tokens = item["reference"].split()
            hyp_tokens = item["hypothesis_tokens"]

            if op == "S":
                operation = "substitution"
                reference_word = ref_tokens[ri]
                hypothesis_word = hyp_tokens[hj]

            elif op == "D":
                operation = "deletion"
                reference_word = ref_tokens[ri]
                hypothesis_word = ""

            elif op == "I":
                operation = "insertion"
                reference_word = ""
                hypothesis_word = hyp_tokens[hj]

            else:
                raise ValueError(op)

            examples.append({
                "dataset": "fsc",
                "reference": item["reference"],
                "hypothesis": record["hypothesis_text"],
                "operation": operation,
                "reference_word": reference_word,
                "hypothesis_word": hypothesis_word,
                "reference_index": (
                    int(ri) if ri is not None else -1
                ),
                "hypothesis_index": (
                    int(hj) if hj is not None else -1
                ),
                "repair_position": int(hpos),
                "repaired_hypothesis": record["repaired_text"],
                "label": label,
                "label_name": (
                    "CONSEQUENTIAL"
                    if label
                    else "NON_CONSEQUENTIAL"
                ),
                "teacher_frame_h": list(h_frame),
                "teacher_frame_repaired": list(repaired_frame),
            })

        return examples

    train_examples = label_hypotheses(
        train_hypotheses,
        "train",
    )

    valid_examples = label_hypotheses(
        valid_hypotheses,
        "valid",
    )

    def deduplicate(rows):
        output = []
        seen = set()

        for row in rows:
            key = (
                row["reference"],
                row["hypothesis"],
                row["operation"],
                row["reference_index"],
                row["hypothesis_index"],
            )

            if key not in seen:
                seen.add(key)
                output.append(row)

        return output

    train_examples = deduplicate(train_examples)
    valid_examples = deduplicate(valid_examples)

    write_jsonl(
        args.output_dir / "train.jsonl",
        train_examples,
    )
    write_jsonl(
        args.output_dir / "valid.jsonl",
        valid_examples,
    )

    def statistics(rows):
        return {
            "examples": len(rows),
            "labels": dict(
                Counter(row["label_name"] for row in rows)
            ),
            "operations": dict(
                Counter(row["operation"] for row in rows)
            ),
            "unique_references": len({
                row["reference"] for row in rows
            }),
            "unique_hypotheses": len({
                row["hypothesis"] for row in rows
            }),
        }

    metadata = {
        "label_definition":
            "q_change = 1[g_A(H) != g_A(H_repaired_edit)]",
        "fsc_train_csv": str(args.fsc_train_csv),
        "seed": args.seed,
        "train_corruptions": args.train_corruptions,
        "valid_corruptions": args.valid_corruptions,
        "student_train_reference_count": len(train_refs),
        "student_valid_reference_count": len(valid_refs),
        "train_valid_reference_overlap": len(
            {x["reference"] for x in train_refs}
            & {x["reference"] for x in valid_refs}
        ),
        "train": statistics(train_examples),
        "valid": statistics(valid_examples),
    }

    (
        args.output_dir / "metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("CONTEXTUAL EDIT-CONSEQUENCE DATASET CREATED")
    print("=" * 80)
    print(json.dumps(metadata, indent=2))
    print("\nSaved:", args.output_dir)


if __name__ == "__main__":
    main()
