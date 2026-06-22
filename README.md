# Hostile Debate Rhetoric and Post-Debate Polls — Replication

**Authors: Bryan Kyung and Katie Kim**

Code and data for the analysis of whether hostile rhetoric in U.S. presidential debates
(2008–2024) moves post-debate polls by party. Every reported number can be reproduced.
The primary rhetorical measure is a **supervised classifier trained on human-coded gold
standards** for all three categories (aggressive, inflammatory, discriminatory); lexicon,
zero-shot, and hate-speech-transformer classifiers are included for robustness.

## Quick reproduce
```bash
pip install numpy pandas
python code/supervised_analysis.py    # trains the 3 supervised classifiers, reports CV F1,
                                       # classifies the 2008–2024 corpus, runs all regressions
```
No GPU, API key, or internet required. The optional transformer/API classifiers
(`run_gpu_classify.py`, `run_hf_validate.py`, `perspective_classify.py`, `hatebert_classify.py`)
need their own hardware/keys and each validate against the human gold before use.

## Structure
```
code/
  supervised_analysis.py            # PRIMARY: train 3 supervised classifiers -> classify -> regress
  train_supervised_discriminatory.py# supervised discriminatory classifier + CV F1 (Section 6.10)
  train_supervised_aggr_infl.py     # supervised aggressive/inflammatory classifiers + CV F1
  score_discriminatory_gold.py      # score any classifier vs the human gold (kappa, IPW P/R/F1)
  reanalysis_master.py              # earlier poll-level regressions (lexicon measure)
  pipeline.py                       # context-aware lexicon classifier
  run_gpu_classify.py               # zero-shot transformer classifier (local GPU)
  run_hf_validate.py                # hate-speech transformer scored vs the 372-sentence gold
  perspective_classify.py           # Google Perspective API (IDENTITY_ATTACK / TOXICITY)
  hatebert_classify.py              # pretrained hate-speech transformer
  RUBRIC.md                         # annotation codebook (definitions + worked examples)
  Program_R2.do                     # Stata version of the reviewer-requested specifications
data/
  panel_2008_2024.csv               # poll panel (14 debates, 554 candidate-poll obs)
  corpus_sentences.csv              # candidate sentences, 2004–2020
  corpus_2024.csv                   # candidate sentences, 2024 debates
  human_gold_372.csv                # 372-sentence double-coded gold — discriminatory
  human_gold_aggr_infl_372.csv      # 372-sentence double-coded gold — aggressive & inflammatory
  discriminatory_coding_key.csv     # strata + sampling weights for the 372-sentence set
  supervised_counts.csv             # supervised discriminatory counts (per debate/party)
  reclassified