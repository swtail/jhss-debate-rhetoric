# Measuring Hostile Rhetoric in U.S. Presidential Debates

Replication materials for a double-blind peer-review submission. Author-identifying information has been removed from code, documentation, and repository metadata.

## Overview

This repository reproduces the measurement validation and polling-association analyses for U.S. general-election presidential debates, 2004–2024. The analysis separates two questions:

- **RQ1, measurement:** Can automated methods identify aggressive, inflammatory, and discriminatory rhetoric against a human-coded benchmark?
- **RQ2, associations and sensitivity:** Conditional on the validated measure, which source-specific associations appear between hostile rhetoric and post-debate polling, and how stable are they under placebo, few-cluster, measurement-error, and specification checks?

## Headline Results

- A supervised classifier trained on a 372-sentence double-coded human gold standard performs best among the tested measures in every category.
- The focal Republican-share placebo test passes under the paper's significance threshold, `p < 0.10`: all six pre-debate Republican-share placebo p-values are above 0.10.
- The cleanest timing result is Democratic-spoken inflammatory rhetoric: it is associated with lower Republican share and has a significant post-minus-placebo difference.
- Democratic-spoken discriminatory rhetoric is also associated with lower Republican share under the validated supervised measure; it is reported as a source-specific association rather than the strongest timing result.
- Content-free placebo treatments, total sentences and non-hostile sentences, do not show comparable associations.

## Repository Structure

```text
code/        Reproduction scripts
data/        Input data and validated analysis panels
figures/     Figures used in the manuscript
results/     Saved text output from main analyses
```

## Main Scripts

Run scripts from the repository root.

```bash
python code/reproduce_supervised_primary.py
python code/classifier_robustness.py
python code/validation_and_robustness.py
python code/sample_size_heterogeneity.py
python code/placebo_validated.py
python code/randomization_inference_validated.py
python code/debate_level_aggregation.py
python code/rq2_sensitivity.py
python code/measurement_error_attenuation.py
python code/placebo_treatment.py
python code/focal_republican_placebo.py
```

## Script Descriptions

- `reproduce_supervised_primary.py`: trains the supervised classifiers, scores the 2004–2024 corpus, builds the validated panel, and estimates pooled and source-specific WLS models.
- `classifier_robustness.py`: compares source-specific coefficients across keyword, context-aware, and supervised measurement instruments.
- `validation_and_robustness.py`: reports cross-validated F1 and wild cluster bootstrap inference.
- `sample_size_heterogeneity.py`: estimates high-sample-size interactions with wild bootstrap inference.
- `placebo_validated.py`: estimates pre-debate placebo tests and formal post-minus-placebo difference tests.
- `randomization_inference_validated.py`: conducts debate-level randomization inference.
- `debate_level_aggregation.py`: collapses to the debate level and re-estimates pooled associations.
- `rq2_sensitivity.py`: runs the 176-specification multiverse and minimum-detectable-association calculations.
- `measurement_error_attenuation.py`: simulates measurement-error attenuation using gold-standard sensitivity and false-positive rates.
- `placebo_treatment.py`: replaces hostile rhetoric with content-free placebo treatments.
- `focal_republican_placebo.py`: reproduces the focal Republican-share post-debate and pre-debate placebo table used to show that all focal placebo p-values exceed 0.10.

## Reproduce

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python code/reproduce_supervised_primary.py
python code/validation_and_robustness.py
python code/placebo_validated.py
python code/placebo_treatment.py
python code/focal_republican_placebo.py
```

The full set of scripts can be run in the order listed above. Some scripts overwrite files in `results/` with regenerated output.

## Data Notes

- `human_gold_aggr_infl_372.csv` and `human_gold_discriminatory_372.csv` contain double-coded validation labels.
- `discriminatory_sampling_key.csv` contains the enriched-sample design and inverse-probability sampling weights.
- `corpus_2004.csv`, `corpus_sentences_2008_2020.csv`, and `corpus_2024.csv` contain sentence-level debate transcripts.
- `panel_2004_2024_validated.csv` is the validated analysis panel produced by the main reproduction script.

## Anonymization Note

Repository metadata and code headers do not identify the authors. Public debate transcript text is preserved as source data and may contain ordinary names spoken during debates; these are not author identifiers.
