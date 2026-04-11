# MyMLLab

A minimal, observable machine learning experimentation lab implemented as a small .NET console app.

## What it does today

- Generates a synthetic linear dataset with controlled Gaussian noise.
- Splits data into train/validation sets.
- Trains a linear regression model with gradient descent.
- Runs a batch experiment over multiple learning rates.
- Logs per-epoch metrics to CSV/JSON for each run.
- Produces a ranked leaderboard by final validation loss.

## Run

```bash
dotnet run --project MyMLLab/MyMLLab.csproj
```

## Output

Running the app creates an `artifacts/` directory with:

- `run_lr_<value>.csv`: epoch-by-epoch training/validation loss and parameter values
- `run_lr_<value>.json`: structured summary + full metric series
- `leaderboard.csv`: ranked experiment summary

## Why this project exists

MyMLLab is intentionally small and explicit so changes are easy to reason about:

- **Transparency:** model parameters and training dynamics are visible.
- **Controlled simplicity:** start with linear models and iterate.
- **Measurable iteration:** each experiment run emits comparable metrics.
- **Experiment-first workflow:** compare configurations side by side.
