# Methodology

This document explains **what each step does, why, and what its limits are**. It matches the four notebooks and `generate_panel_data.py` in this repository. Numbers come from the delivered data (seed 42); other machines may differ in the last decimals.

> **Synthetic data, demonstration only.** The data contains no real company or customer information, and every relationship in it is built into the generator. Results show that the pipeline works, not that the hypotheses hold in a real customer base.

## Contents
1. [Problem and definitions](#1-problem-and-definitions)
2. [The synthetic data](#2-the-synthetic-data)
3. [Cleaning](#3-cleaning)
4. [Timeline analysis](#4-timeline-analysis)
5. [Feature engineering](#5-feature-engineering)
6. [Stratified splitting](#6-stratified-splitting)
7. [Modelling and tuning](#7-modelling-and-tuning)
8. [Evaluation](#8-evaluation)
9. [Explainability](#9-explainability)
10. [The cost-based decision](#10-the-cost-based-decision)
11. [Export](#11-export)
12. [Validation performed](#12-validation-performed)
13. [Limitations and next steps](#13-limitations-and-next-steps)

---

## 1. Problem and definitions

**Value compression.** A customer stays on the network but spends much less. Standard churn triggers watch for cancelled lines, so they miss this group.

**Hypothesis behind the project.** Some customers who are about to leave first give up quietly after repeated problems that were never resolved. If they behave like customers who have *already* left, they can be caught earlier from behaviour alone.

**The label** (notebook 01, section 2). For each customer, take the last three observed months and the three months before them:

```
change_rev = (recent average revenue - earlier average revenue) / earlier average revenue * 100
value_compressed_flag = 1  if change_rev <= -25
```

Active customers are observed to month 12 (months 10–12 against 7–9). Churned customers are observed to the month before they left.

**Four segments** (churn status × label): Active-Normal (155,582), Active-Compressed (19,989), Churn-Normal (19,095), Churn-Compressed (5,334).

**Who is modelled.** Active customers only (175,571), because they can still be contacted. Churned customers are used for the timeline comparison.

**Prediction setup.** Features come from months 1–9 only; the label is built from months 10–12 (and 7–9 as the baseline). Revenue defines the label and is **never** a feature.

```
Month:    1  2  3  4  5  6  7  8  9 | 10  11  12
          [------ features -------]   [ label window ]
```

## 2. The synthetic data

`generate_panel_data.py` writes two tables:

| Table | Rows | Content |
|---|---:|---|
| `panel_monthly.parquet` | 2,352,248 | one row per active customer-month: revenue, overage fee, data-overage fee, dropped/blocked calls, support calls, tenure |
| `customers.parquet` | 200,000 | static facts: income bracket (−1 = unknown), churn flag and month, plus `latent_*` columns |

**Design choices (all built in):**
- Three hidden groups: stable (74.6%), compressing (13.2%), churning (12.2%).
- Compressing and churning customers share **one deterioration process**: for 4–6 months before their event, support calls, dropped calls and overage fees rise. Churners get a slightly stronger amplitude.
- Churners leave in months 10–12; compression starts in months 10–11.
- Imperfections so the task is not trivial: 15% of at-risk customers show weak signals, and 12% of stable customers have a temporary false-alarm episode that recovers.
- Revenue has mild seasonality; overage fees are part of revenue.
- The `latent_*` columns are the generator's **answer key**. They exist only because the data is synthetic and are removed in notebook 01 so they can never leak into a model.

## 3. Cleaning

Notebook 01, section 1. The checks: data types, missing values, duplicate customer-months, impossible (negative) values, month range, and the business rule *"no rows on or after a customer's churn month"*. Unknown income is kept as −1 and flagged with `income_missing`, so tree models can use the missingness.

This data is clean by construction, so the checks confirm it. With real data this stage would also repair what it finds.

## 4. Timeline analysis

Notebook 01, section 3. For each of the four segments, the monthly average of support calls, dropped calls, overage fee and revenue is plotted, then compared in months 7–9 as a ratio to the Active-Normal level.

**What the run shows.** All three at-risk segments (Active-Compressed, Churn-Normal, Churn-Compressed) rise together from about month 5. In months 7–9 their support calls and dropped calls are about **1.3–1.8×** the normal level. Overage differs more between segments (about 1.1–2.0×) because overage is itself part of the revenue that defines the label. The share of active customers who compress rises steadily across support-call quintiles (about 8% to 19%).

**What it does not show.** The generator builds the shared process in, so the similarity is expected. On real data, this comparison is the real test of the hypothesis, and it could fail.

## 5. Feature engineering

Notebook 02. For each of four behaviour columns (overage fee, data-overage fee, dropped/blocked calls, support calls), per active customer over **months 1–9**:

| Suffix | Meaning |
|---|---|
| `_mean` | average of months 1–9 |
| `_first3` | average of months 1–3 (the customer's early normal) |
| `_last3` | average of months 7–9 (most recent behaviour) |
| `_max` | worst single month |
| `_change` | `_last3` − `_first3` (a simple trend) |

Plus `tenure_months` (at month 9), `estimated_income_bracket` and `income_missing`: **23 features** in total.

**Leakage rules, checked in code:** no revenue-based column, no `latent_*`, nothing containing "churn" or "segment", and only months 1–9.

**Do the trend features help? (ablation, notebook 03).** Validation AUC with all 23 features is 0.838; **without** the four `_change` features it is 0.839; with **only** them it is 0.815. The `_change` features rank highest in importance (about 72% of total gain) but add nothing the model needs, because `_last3` and `_first3` stay in the table. This is why the documentation does not claim that the trend engineering improved accuracy here.

## 6. Stratified splitting

Notebook 02, section 4. Compressed customers are a minority and customer types differ (new vs long-standing, income level, unknown income). A plain random split can leave some types under-represented in one part. Stratifying keeps the same mix of customer types in every part.

- **Strata:** label × income group (missing / low / mid / high) × tenure band (new / mid / long) = **24 customer types**.
- **Split:** 30% test; the rest split 80% train / 20% validation. Sizes: train 98,319, validation 24,580, test 52,672.
- **Checked:** label rate, unknown-income share and long-tenure share match across the three sets to within a hundredth of a percentage point, and no customer appears in two sets.

This is a **stratified hold-out**, not k-fold cross-validation. Validation is used to tune and compare; the test set is used **once**, at the end.

## 7. Modelling and tuning

Notebook 03.

| Model | Role |
|---|---|
| Logistic regression (with scaling) | simple, explainable baseline |
| Decision tree, depth 4 | readable rules |
| LightGBM, default | strong tabular baseline |
| **LightGBM, tuned with Optuna** | champion |

**Optuna.** 20 trials (TPE sampler, seed 42). Each trial trains LightGBM on the training set and scores AUC on the validation set. Search space: `n_estimators` 100–500, `learning_rate` 0.01–0.2 (log), `num_leaves` 8–64, `min_child_samples` 20–200. Best: 395 trees, learning rate 0.036, 22 leaves, minimum 172 samples per leaf. It improved validation AUC by only about 0.003 over default LightGBM.

**Class weights are left at the default** (`scale_pos_weight` is not tuned). Class weighting mostly rescales probabilities, and this project needs probabilities that mean what they say, because the cost rule in section 10 relies on them. The cost-based threshold handles the imbalance instead.

**Champion choice** uses validation AUC, never test.

## 8. Evaluation

Notebook 04, section 1, on the test set.

| Model | Validation AUC | Test AUC |
|---|---:|---:|
| Decision tree | 0.766 | 0.769 |
| Logistic regression | 0.830 | 0.832 |
| LightGBM, default | 0.838 | 0.838 |
| LightGBM, tuned | 0.841 | 0.841 |

AUC measures how well a model *ranks* customers (0.5 = guessing, 1.0 = perfect). The notebook also shows ROC curves, confusion matrices at 0.50, and a **calibration curve**: predicted probabilities track the real compression rate closely, which supports the cost rule.

At the default 0.50 cut-off every model catches only a small share of compressed customers (low recall), which is the reason section 10 exists.

## 9. Explainability

Notebook 04, section 2: LightGBM feature importance ("gain"), SHAP bar and beeswarm plots on 2,000 test customers, a SHAP waterfall for the highest-risk customer in that sample, and average predicted risk by overage level.

**Read with care.** SHAP shows each feature's effect *given the other features*. Features carrying the same information (`_change`, `_last3`, `_first3`) share the credit, and a feature can point in an unexpected direction (for example `monthly_overage_fee_mean`). SHAP describes *how the model reasons*, not cause and effect.

## 10. The cost-based decision

Notebook 04, section 3.

**Economics (assumptions, editable).** V = 500 EGP kept when a compressing customer is retained; C = 50 EGP per offer.

```
net value(θ) = TP(θ) × V − (TP(θ) + FP(θ)) × C
break-even probability = C / V = 0.10
```

A customer is worth contacting when `chance × V > C`. This holds when probabilities are well calibrated.

**Choosing θ without cheating.** Thresholds 0.01–0.99 are scanned on the **validation** set; the best is **0.09**, next to the 0.10 break-even. The **test** set is used only to report. Results on the test set (52,672 customers):

| Threshold | Flagged | TP | FP | FN | Net value (EGP) |
|---|---:|---:|---:|---:|---:|
| 0.50 | 2,523 | 1,610 | 913 | 4,387 | 678,850 |
| 0.10 (break-even) | 15,360 | 4,560 | 10,800 | 1,437 | 1,512,000 |
| 0.09 (chosen) | 16,695 | 4,700 | 11,995 | 1,297 | 1,515,250 |

At 0.09: precision 28%, recall 78%, and net value about 2.2× the default (+836,400 EGP).

**Sensitivity to the "every caught customer is saved" assumption.** If only part of the caught customers accept the offer, net value at the same threshold is **340,250 EGP at 50%** and **−247,250 EGP at 25%**. (A lower save rate would also raise the best threshold; this is not re-optimised.) The economics in this project therefore depend on a quantity the model cannot measure. A deployment would estimate it with a randomized control group.

## 11. Export

`work/retention_model.joblib` is one dictionary: the model, the 23-feature list, the chosen threshold, V and C, the LightGBM and scikit-learn versions, and a note that it was trained on synthetic data. Notebook 04 reloads it, confirms identical predictions, and scores five "new" customers (risk and "send offer" / "no offer"). `work/test_predictions.parquet` stores the test-set scores and flags.

## 12. Validation performed

- All four notebooks execute from a clean folder with no errors or warnings (16 plots).
- The data regenerates identically from the seed.
- No leakage columns among the features; train, validation and test share no customer, and together equal the active customers.
- The label was rebuilt independently from the raw data and matches for every customer except one on the cut-off (a `float32` rounding effect, 0.0005%).
- The money figures were recomputed independently from the saved predictions and match the notebook.
- The joblib model, loaded in a brand-new Python process, reproduces the saved scores and flags exactly.
- No figure from any confidential source appears in the repository.

Not covered: Colab and Kaggle paths, other library versions, and any real data.

## 13. Limitations and next steps

- **Synthetic by construction.** The shared deterioration process, the false alarms and the weak-signal customers are generator choices.
- **Label design.** The label uses a ≤ −25% revenue change; about a quarter of the generator's compressing customers do not reach it inside the window, and a few stable customers cross it by noise.
- **Correlation, not response.** The model predicts who will compress, not who will respond to an offer; uplift modelling and a control group would be the next step.
- **Calibration** was checked on one dataset and should be re-checked on any new one.
