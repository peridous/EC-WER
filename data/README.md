# Datasets

FSC (Fluent Speech Commands), SLURP, and SLUE-VoxPopuli are not
redistributed in this repository. Obtain each dataset from its official
distributor and follow its license and access conditions.

Keep downloaded datasets, audio, transcripts, generated supervision, and
local preprocessing outputs outside the public release. Files in this
directory other than this README are ignored by Git.

The FSC supervision builder requires the official training CSV with
`transcription`, `action`, `object`, and `location` columns. It trains a
frozen text evaluator on FSC training data, retains evaluator-correct
commands, and splits unique commands before generating corruptions.

SLURP and SLUE-VoxPopuli are transfer evaluation datasets. The released
scorer accepts locally prepared CSVs containing `reference` and
`hypothesis`; an optional binary `downstream_failure` column enables
AUROC/AUPRC evaluation. Supply hypotheses and failure labels from your
chosen ASR systems and downstream evaluators. This minimal release does
not generate those ASR outputs or train the transfer-task evaluators.

See the repository README for FSC preparation, training, and scoring
commands. Generated JSONL and CSV files are ignored throughout the repo.
