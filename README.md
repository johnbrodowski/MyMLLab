# MyMLLab

A minimal but feature-rich, fully observable machine learning experimentation lab in a single .NET console app.

## Included capabilities

- Synthetic linear dataset generation with deterministic seed and configurable Gaussian noise.
- Train/validation split with optional feature normalization using train-only statistics.
- Linear regression training with:
  - SGD optimizer
  - Adam optimizer
  - L2 regularization
  - Early stopping
- Experiment sweeps over optimizer, learning rate, and L2 penalty.
- Per-epoch logging of:
  - training loss
  - validation loss
  - weight
  - bias
  - gradient norm
- Prediction error distribution metrics (mean, stddev, p50, p90).
- Ranked leaderboard across all runs.

## Run

```bash
dotnet run --project MyMLLab/MyMLLab.csproj
```

## Artifacts

The app writes to `artifacts/`:

- `run_opt_<optimizer>_lr_<lr>_l2_<l2>.csv`
- `run_opt_<optimizer>_lr_<lr>_l2_<l2>.json`
- `leaderboard.csv`

All outputs are designed to support side-by-side analysis and reproducible experimentation.
