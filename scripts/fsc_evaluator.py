from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FRAME_COLUMNS = ['action', 'object', 'location']
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_tokens(text: object) -> list[str]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    return TOKEN_RE.findall(str(text).lower().replace("’", "'"))


def normalize_text(text: object) -> str:
    return " ".join(normalize_tokens(text))


def make_classifier(seed: int) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=100000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=4000,
                random_state=seed,
            ),
        ),
    ])


def build_fsc_resources(train_csv: Path, seed: int):
    frame = pd.read_csv(train_csv)
    required = {"transcription", *FRAME_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"FSC training CSV missing {sorted(missing)}")

    frame = frame.copy()
    frame["_text"] = frame["transcription"].map(normalize_text)
    frame = frame[frame["_text"].str.len() > 0].copy()

    targets: dict[str, tuple[str, str, str]] = {}
    ambiguous: set[str] = set()
    for values in frame[["_text", *FRAME_COLUMNS]].itertuples(index=False, name=None):
        text = str(values[0])
        target = tuple(str(value) for value in values[1:])
        if text in targets and targets[text] != target:
            ambiguous.add(text)
        else:
            targets[text] = target
    for text in ambiguous:
        targets.pop(text, None)

    evaluators: dict[str, Pipeline] = {}
    for offset, column in enumerate(FRAME_COLUMNS):
        evaluator = make_classifier(seed + offset)
        evaluator.fit(frame["_text"], frame[column].astype(str))
        evaluators[column] = evaluator
    return targets, evaluators


def fsc_predict_texts(
    texts: list[str],
    evaluators: dict[str, Pipeline],
) -> dict[str, dict[str, Any]]:
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    output = {text: {"pred": {}, "probs": {}} for text in unique}
    for column in FRAME_COLUMNS:
        evaluator = evaluators[column]
        predicted = evaluator.predict(unique)
        probabilities = evaluator.predict_proba(unique)
        classes = [str(value) for value in evaluator.named_steps["classifier"].classes_]
        for text, pred, probs in zip(unique, predicted, probabilities):
            output[text]["pred"][column] = str(pred)
            output[text]["probs"][column] = {
                label: float(probs[index])
                for index, label in enumerate(classes)
            }
    return output


def fsc_prediction(result: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(result["pred"][column]) for column in FRAME_COLUMNS)

