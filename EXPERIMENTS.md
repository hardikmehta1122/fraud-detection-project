# Experiment log

Same split and feature set throughout; only the model config and
imbalance-handling strategy vary. Winner picked by validation F1.

| config | max_depth | eta | imbalance handling | val F1 | test acc | test prec | test recall | test F1 | test AUC |
|---|---|---|---|---|---|---|---|---|---|
| baseline_balanced | 6 | 0.08 | scale_pos_weight | 0.9664 | 0.9976 | 0.9727 | 0.9659 | 0.9693 | 0.9997 |
| shallow_trees | 3 | 0.08 | scale_pos_weight | 0.9656 | 0.9977 | 0.9689 | 0.9723 | 0.9706 | 0.9997 |
| deep_slow_learn | 9 | 0.03 | scale_pos_weight | 0.9663 | 0.9975 | 0.9699 | 0.9665 | 0.9682 | 0.9997 |
| no_imbalance_handling (winner) | 6 | 0.08 | none | 0.9677 | 0.9975 | 0.9824 | 0.9535 | 0.9677 | 0.9997 |
| oversample_minority | 6 | 0.08 | oversampling | 0.9661 | 0.9974 | 0.9801 | 0.9553 | 0.9675 | 0.9997 |

## Notes

- `no_imbalance_handling` trains without `scale_pos_weight` or resampling.
- `oversample_minority` duplicates minority rows in the training split as an
  alternative to loss reweighting.
- `shallow_trees` and `deep_slow_learn` bracket the baseline's tree capacity.
- All five configs fall within ~0.002 validation F1 of each other, so config
  choice has little effect on this dataset.

Regenerate with `python -m model.experiments`.
