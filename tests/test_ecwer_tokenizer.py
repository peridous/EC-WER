import copy
import ast
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ecwer import load_ecwer_tokenizer


class TrainingTokenizerConfigurationTest(unittest.TestCase):
    def test_training_explicitly_preserves_metaspace(self):
        path = Path(__file__).resolve().parents[1] / "scripts/train_edit_conditioned_classifier.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "AutoTokenizer"
                 and node.func.attr == "from_pretrained"]
        self.assertEqual(len(calls), 1)
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
        self.assertIs(settings["fix_mistral_regex"], False)


class TokenizerRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = os.environ.get("ECWER_TOKENIZER_PATH", "microsoft/deberta-v3-small")
        cls.tokenizer = load_ecwer_tokenizer(cls.source)

    def test_training_ids(self):
        text = (
            "[REFERENCE] turn on the light [HYPOTHESIS] turn off the light "
            "[EDIT_TYPE] substitution [REFERENCE_EDIT] on [HYPOTHESIS_EDIT] off"
        )
        self.assertEqual(self.tokenizer(text)["input_ids"], [
            1, 647, 62740, 6347, 29764, 592, 930, 277, 262, 731,
            647, 41826, 16827, 17102, 35693, 592, 930, 442, 262, 731,
            647, 71648, 616, 63447, 592, 17433, 647, 62740, 6347,
            29764, 616, 71648, 592, 277, 647, 41826, 16827, 17102,
            35693, 616, 71648, 592, 442, 2,
        ])

    def test_validation_encodings(self):
        validation = os.environ.get("ECWER_VALID_JSONL")
        original = os.environ.get("ECWER_ORIGINAL_TOKENIZER_JSON")
        if not validation or not original:
            self.skipTest("Set ECWER_VALID_JSONL and ECWER_ORIGINAL_TOKENIZER_JSON for full regression")
        texts = [json.loads(line)["text"] for line in
                 Path(validation).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(texts), 3626)
        # Read the original training backend directly, bypassing AutoTokenizer's
        # environment-dependent regex migration and current loader settings.
        reference = Tokenizer.from_file(original)
        reference.no_padding()
        reference.enable_truncation(max_length=128)
        expected = [encoding.ids for encoding in reference.encode_batch(texts)]
        actual = self.tokenizer(texts, truncation=True, max_length=128)["input_ids"]
        mismatches = sum(left != right for left, right in zip(expected, actual))
        self.assertEqual(len(actual), len(expected))
        mode = "offline" if os.environ.get("HF_HUB_OFFLINE") == "1" else "online"
        print(f"{mode}: {len(texts) - mismatches:,} / {len(texts):,} validation encodings match; "
              f"{mismatches} mismatches")
        self.assertEqual(mismatches, 0)

    def test_changed_preprocessing_is_rejected(self):
        changed = copy.deepcopy(self.tokenizer)
        changed.backend_tokenizer.pre_tokenizer = Whitespace()
        with patch("evaluate_ecwer.AutoTokenizer.from_pretrained", return_value=changed):
            with self.assertRaisesRegex(RuntimeError, "training token IDs"):
                load_ecwer_tokenizer(self.source)

    def test_untested_version_is_rejected(self):
        with patch("evaluate_ecwer.version", return_value="0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "requires transformers"):
                load_ecwer_tokenizer(self.source)


if __name__ == "__main__":
    unittest.main()
