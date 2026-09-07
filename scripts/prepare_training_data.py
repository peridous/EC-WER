from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsc_evaluator import build_fsc_resources, fsc_predict_texts, fsc_prediction


def build_input_text(reference, hypothesis, operation, reference_word, hypothesis_word):
    return " ".join([
        "[REFERENCE]", reference, "[HYPOTHESIS]", hypothesis,
        "[EDIT_TYPE]", operation, "[REFERENCE_EDIT]", reference_word or "[EMPTY]",
        "[HYPOTHESIS_EDIT]", hypothesis_word or "[EMPTY]",
    ])


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Format binary edit labels and construct matched transcript-failure pairs."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--edit_output_dir", type=Path, required=True)
    parser.add_argument("--direct_output_dir", type=Path, required=True)
    parser.add_argument("--fsc_train_csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    paths = [p.resolve() for p in (
        args.input_dir, args.edit_output_dir, args.direct_output_dir
    )]
    if len(set(paths)) != 3:
        raise ValueError("Input and output directories must be distinct")
    splits = {s: load_jsonl(args.input_dir / f"{s}.jsonl") for s in ("train", "valid")}
    refs = {s: {r["reference"] for r in rows} for s, rows in splits.items()}
    if refs["train"] & refs["valid"]:
        raise ValueError("Train and validation references overlap")
    _, evaluator = build_fsc_resources(args.fsc_train_csv, args.seed)
    predictions = fsc_predict_texts(sorted(refs["train"] | refs["valid"]), evaluator)
    for split, rows in splits.items():
        edits, pairs = [], {}
        for row in rows:
            label = row["label"]
            expected = int(row["teacher_frame_h"] != row["teacher_frame_repaired"])
            if label not in (0, 1) or label != expected:
                raise ValueError("Edit label does not equal binary teacher-frame change")
            edits.append({"text": build_input_text(
                row["reference"], row["hypothesis"], row["operation"],
                row["reference_word"], row["hypothesis_word"],
            ), "label": label})
            key = (row["reference"], row["hypothesis"])
            reference_frame = list(fsc_prediction(predictions[row["reference"]]))
            hypothesis_frame = row["teacher_frame_h"]
            failure = int(reference_frame != hypothesis_frame)
            pair = {
                "reference": key[0], "hypothesis": key[1],
                "teacher_frame_reference": reference_frame,
                "teacher_frame_hypothesis": hypothesis_frame,
                "label": failure,
                "label_name": "FAILURE" if failure else "NO_FAILURE",
            }
            if key in pairs and pairs[key] != pair:
                raise ValueError("Conflicting teacher predictions for a transcript pair")
            pairs[key] = pair
        save_jsonl(args.edit_output_dir / f"{split}.jsonl", edits)
        save_jsonl(args.direct_output_dir / f"{split}.jsonl",
                   [pairs[key] for key in sorted(pairs)])
        print(f"{split}: {len(edits)} edits; {len(pairs)} Direct Estimate pairs")


if __name__ == "__main__":
    main()
