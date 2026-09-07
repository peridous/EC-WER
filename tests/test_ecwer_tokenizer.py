import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ecwer import load_ecwer_tokenizer


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
