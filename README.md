# Hostile Debate Rhetoric and Post-Debate Polls — Replication

**Authors: Bryan Kyung and Katie Kim**

Validated, multi-classifier re-analysis of whether hostile rhetoric in U.S. presidential
debates (2004–2020) moves post-debate polls asymmetrically by party. This repository
contains all data and code, so every reported number can be reproduced.
It also provides the materials the reviewer requested (transparent classifier code,
annotation codebook, validation, and the full dataset — replacing "available upon request").

## Structure
```
code/
  reanalysis_master.py     # reproduces every regression (numpy/pandas only, no scipy)
  pipeline.py              # context-aware classifier -> variables -> models
  Program_R2.do            # Stata version of the reviewer-requested specifications
  RUBRIC.md                # annotation codebook (definitions + worked examples)
  run_gpu_classify.py      # zero-shot transformer classifier (local GPU)
  perspective_classify.py  # Google Perspective API (IDENTITY_ATTACK / TOXICITY)
  hatebert_classify.py     # pretrained hate-speech transformer
data/
  poll_final_data_with_change.dta   # analysis panel (463 candidate-poll obs, 12 debates)
  corpus_sentences.csv              # ~24,500 candidate sentences
  gold_sample.json                  # 60-sentence expert gold standard (validation)
  reclassified_counts_v2.csv        # transparent-classifier counts
  zeroshot_counts.csv               # GPU zero-shot counts
results/
  RESULTS.txt                       # full regression output
  comparison.txt                    # four-classifier comparison
```

## Reproduce
```bash
pip install numpy pandas
python code/reanalysis_master.py     # prints all regression results from the .dta
```
The transformer and API classifiers (GPU zero-shot, Perspective, HateBERT) require their own keys/hardware;
each script documents its setup and validates on `data/gold_sample.json` before use.

## Key finding
Across four independent, gold-validated classifiers the predicted asymmetric "civility
penalty" does not reproduce: aggressive rhetoric is inert, discriminatory rhetoric is too
rare to measure, and the inflammatory association is small and changes sign with the
classifier, the estimator (OLS vs. IV), and the unit of analysis. We report a validated null.

## Validation summary (vs. 60-sentence expert gold standard)
| Classifier | Inflammatory F1 | Aggressive F1 | Discriminatory |
|---|---|---|---|
| Plain keyword | 0.77 | 0.11 | 0 in gold; unstable |
| Context-aware | 0.77 | 0.11 | 0 in gold; 8 flagged |
| Zero-shot DeBERTa | 0.45 | 0.29 | 0 in gold; 1,188 flagged |
