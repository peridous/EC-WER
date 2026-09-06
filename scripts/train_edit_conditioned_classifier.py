from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = [
    "NON_CONSEQUENTIAL",
    "CONSEQUENTIAL",
]

ID2LABEL = {
    index: label
    for index, label in enumerate(LABELS)
}

LABEL2ID = {
    label: index
    for index, label in ID2LABEL.items()
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if "text" not in row or "label" not in row:
                raise RuntimeError(
                    f"{path}:{line_number}: missing text/label"
                )

            label = int(row["label"])

            if label not in {0, 1}:
                raise RuntimeError(
                    f"{path}:{line_number}: bad label {label}"
                )

            rows.append({
                "text": str(row["text"]),
                "label": label,
            })

    return rows


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: torch.Tensor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        loss = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(
                outputs.logits.device
            ),
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def probabilities_from_logits(
    logits: np.ndarray,
) -> np.ndarray:
    logits = logits - logits.max(
        axis=-1,
        keepdims=True,
    )
    exponentials = np.exp(logits)

    return exponentials / exponentials.sum(
        axis=-1,
        keepdims=True,
    )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = probabilities_from_logits(
        np.asarray(logits)
    )

    positive = probabilities[:, LABEL2ID["CONSEQUENTIAL"]]
    predictions = np.argmax(
        probabilities,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "auprc": average_precision_score(
            labels,
            positive,
        ),
        "auroc": roc_auc_score(
            labels,
            positive,
        ),
        "brier": float(
            np.mean(
                (
                    positive
                    - np.asarray(labels)
                )
                ** 2
            )
        ),
    }


def make_training_arguments(
    args,
) -> TrainingArguments:
    kwargs = {
        "output_dir": str(
            args.output_dir / "checkpoints"
        ),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": (
            args.batch_size
        ),
        "per_device_eval_batch_size": (
            args.batch_size
        ),
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "auprc",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": torch.cuda.is_available(),
        "remove_unused_columns": True,
    }

    signature = inspect.signature(
        TrainingArguments.__init__
    )

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--valid_jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="microsoft/deberta-v3-small",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260807,
    )

    args = parser.parse_args()
    set_seed(args.seed)

    train_rows = load_jsonl(args.train_jsonl)
    valid_rows = load_jsonl(args.valid_jsonl)

    print("Train examples:", f"{len(train_rows):,}")
    print("Valid examples:", f"{len(valid_rows):,}")

    train_counts = Counter(
        int(row["label"])
        for row in train_rows
    )

    total = sum(train_counts.values())

    class_weights = torch.tensor(
        [
            total / (
                len(LABELS)
                * train_counts[index]
            )
            for index in range(len(LABELS))
        ],
        dtype=torch.float32,
    )

    print("\nTraining class counts:")

    for index, label in ID2LABEL.items():
        print(
            f"  {label:22s}"
            f"{train_counts[index]:,}"
        )

    print("\nClass weights:")

    for index, label in ID2LABEL.items():
        print(
            f"  {label:22s}"
            f"{class_weights[index]:.6f}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        fix_mistral_regex=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    )

    train_dataset = Dataset.from_list(
        train_rows
    )
    valid_dataset = Dataset.from_list(
        valid_rows
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing train",
    )

    valid_dataset = valid_dataset.map(
        tokenize,
        batched=True,
        desc="Tokenizing valid",
    )

    train_dataset = train_dataset.rename_column(
        "label",
        "labels",
    )
    valid_dataset = valid_dataset.rename_column(
        "label",
        "labels",
    )

    train_dataset = train_dataset.remove_columns(
        ["text"]
    )
    valid_dataset = valid_dataset.remove_columns(
        ["text"]
    )

    trainer = WeightedTrainer(
        model=model,
        args=make_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.train()

    metrics = trainer.evaluate()

    best_dir = args.output_dir / "best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    serializable_metrics = {
        key: (
            float(value)
            if isinstance(
                value,
                (int, float, np.number),
            )
            else value
        )
        for key, value in metrics.items()
    }

    (
        args.output_dir
        / "validation_metrics.json"
    ).write_text(
        json.dumps(
            serializable_metrics,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "model_name": args.model_name,
        "labels": LABELS,
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "input": (
            "reference + hypothesis + aligned edit"
        ),
        "prediction_target": (
            "binary aligned-edit consequence"
        ),
    }

    (
        args.output_dir
        / "training_config.json"
    ).write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFinal validation metrics:")

    for key, value in sorted(
        serializable_metrics.items()
    ):
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nSaved model:", best_dir)


if __name__ == "__main__":
    main()
