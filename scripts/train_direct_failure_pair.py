
import argparse
import inspect
import json
from pathlib import Path

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


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "reference": str(r["reference"]),
                "hypothesis": str(r["hypothesis"]),
                "label": int(r["label"]),
            })
    return rows


def probabilities(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(pred):
    logits, labels = pred
    if isinstance(logits, tuple):
        logits = logits[0]

    p = probabilities(np.asarray(logits))[:, 1]
    yhat = (p >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, yhat),
        "f1": f1_score(labels, yhat, zero_division=0),
        "precision": precision_score(labels, yhat, zero_division=0),
        "recall": recall_score(labels, yhat, zero_division=0),
        "auprc": average_precision_score(labels, p),
        "auroc": roc_auc_score(labels, p),
        "brier": float(np.mean((p - np.asarray(labels)) ** 2)),
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
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
            weight=self.class_weights.to(outputs.logits.device),
        )

        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--valid_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_jsonl)
    valid_rows = load_jsonl(args.valid_jsonl)

    print("Train:", len(train_rows))
    print("Valid:", len(valid_rows))

    train_labels = np.array([r["label"] for r in train_rows])
    counts = np.bincount(train_labels, minlength=2)

    class_weights = torch.tensor(
        len(train_labels) / (2.0 * counts),
        dtype=torch.float32,
    )

    print("Train counts:", counts.tolist())
    print("Class weights:", class_weights.tolist())

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NO_FAILURE", 1: "FAILURE"},
        label2id={"NO_FAILURE": 0, "FAILURE": 1},
    )

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    def tokenize(batch):
        enc = tokenizer(
            batch["reference"],
            batch["hypothesis"],
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    train_ds = train_ds.map(
        tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        tokenize,
        batched=True,
        remove_columns=valid_ds.column_names,
    )

    kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.10,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=True,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**kwargs)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    metrics = trainer.evaluate()

    best_dir = output_dir / "best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print("\nFinal validation metrics:")
    for k in sorted(metrics):
        if isinstance(metrics[k], (int, float)):
            print(f"{k}: {metrics[k]:.6f}")

    print("\nSaved model:", best_dir)


if __name__ == "__main__":
    main()
