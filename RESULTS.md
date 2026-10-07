# Results

## Test metrics

Held-out test split (n = 42,722), not used during training or threshold
selection.

| metric | value |
|---|---|
| accuracy | 99.7% |
| precision (fraud) | 98.2% |
| recall (fraud) | 95.4% |
| F1 (fraud) | 96.8% |
| ROC-AUC | 0.9997 |

Config selected by the validation-F1 sweep in [EXPERIMENTS.md](EXPERIMENTS.md).

## Methodology

- Records are split into train/validation/test before any statistic is
  computed.
- Split-dependent features (per-category amount statistics) are fit on the
  training split only.
- The decision threshold is chosen on validation and applied once to test.
- Per-prediction explanations use XGBoost's `pred_contribs` (tree SHAP).

## Leakage checks

Two checks quantify the effect of common pipeline mistakes. Both run on the
same split and report the delta against the correct pipeline.

**Pre-split statistics** (`model/leakage_demo.py`) — fitting per-category
statistics on the full dataset before splitting:

| metric | leaky | correct | delta |
|---|---|---|---|
| accuracy | 0.9975 | 0.9975 | +0.0000 |
| precision | 0.9661 | 0.9824 | -0.0163 |
| recall | 0.9723 | 0.9535 | +0.0188 |
| F1 | 0.9692 | 0.9677 | +0.0015 |
| ROC-AUC | 0.9997 | 0.9997 | +0.0000 |

The effect is small here because `merchant_category` is a coarse key with
many rows per group in both splits, so the statistic is stable. The effect
is larger for identity- or time-correlated keys (per-account history,
non-causal rolling windows).

**Unvalidated training data** (`model/validation_timing_leakage_demo.py`) —
including the malformed records ingestion would reject in the training split:

| metric | with malformed rows | clean | delta |
|---|---|---|---|
| accuracy | 0.9977 | 0.9975 | +0.0002 |
| precision | 0.9819 | 0.9824 | -0.0005 |
| recall | 0.9588 | 0.9535 | +0.0053 |
| F1 | 0.9702 | 0.9677 | +0.0025 |

With ~1.7% of training rows carrying scattered field-level corruption but
correct labels, the effect is negligible. It grows with the corrupted
fraction, label corruption, or loss of predictive fields.

## Data

The dataset is synthetic (`data/generate_data.py`): fraud and legitimate
records are drawn from distinct feature distributions, which makes the
classes more separable than real transaction data. Metrics reflect that
and are not a benchmark claim. The document layouts in the dashboard are
generated mock-ups, not OCR output.
