# MyMLLab

<img width="883" height="253" alt="image" src="https://github.com/user-attachments/assets/aea5c8d1-a5fa-4a29-be60-a9fb43ec324f" />


A minimal, purely C#, dependency-free machine learning experimentation lab. 

MyMLLab builds a complete Machine Learning training pipeline from absolute scratch. Instead of hiding the math behind massive libraries like PyTorch or ML.NET, this project implements the fundamental calculus and statistics required to train a model in plain, readable C#. It is designed as an educational sandbox to help developers understand exactly how models learn.

## What does it do?
The lab focuses on **Linear Regression** (fitting a line to data using $y = wx + b$). 
1. **Generates** a "fake" dataset based on a secret mathematical rule, adding random real-world noise to make it messy.
2. **Splits** the data into a "Training" set (which the model is allowed to study) and a "Validation" set (which the model is completely blind to, used as a final exam).
3. **Sweeps** through dozens of combinations of algorithms and configurations (hyperparameters) to see which combination discovers the secret mathematical rule the fastest and most accurately.

## Included Capabilities

- **Synthetic Data Generation**: Deterministic seeded dataset creation with configurable Gaussian noise and bounds.
- **Data Preprocessing**: Train/validation splitting with optional feature normalization (Z-score scaling) using strictly train-only statistics to prevent data leakage.
- **Training Engine**:
  - **Optimizers**: Includes both standard Stochastic Gradient Descent (**SGD**) and the adaptive, momentum-based **Adam** optimizer.
  - **Regularization**: **L2 Penalty** (Ridge Regression) to punish the model for relying too heavily on large weights, preventing overfitting.
  - **Early Stopping**: Automatically halts training if the model stops improving, rewinding to the best historical checkpoint.
- **Experiment Orchestration**: "Grid Search" sweeping over multiple learning rates, optimizers, and L2 penalties simultaneously.
- **Rich Observability**:
  - Per-epoch tracking of training loss, validation loss, weight, bias, and gradient norm.
  - Prediction error distribution metrics (mean, stddev, 50th percentile, 90th percentile).
  - Automated convergence diagnostics (`stable`, `plateau`, `still-learning`, `underfit`, `overfit`).

## Running the Lab

Ensure you have the .NET SDK installed, then run:

```bash
dotnet run --project MyMLLab/MyMLLab.csproj
```

### Things to Try
To get the most out of this lab, open `Program.cs` and try tweaking the `DatasetDefinition` and `ExperimentDefinition`:
* **Increase the `NoiseStdDev`**: Make the data messier and watch how the model struggles to find the exact slope.
* **Remove `NormalizeFeature`**: Set this to `false` and watch the SGD optimizer fail to converge (or require drastically smaller learning rates).
* **Add a massive `L2Penalty`**: Watch the model deliberately choose a bad slope because the penalty for a large weight outweighed the reward for being accurate.

## Understanding the Artifacts

When the run completes, the app writes an extensive set of data to the `artifacts/` folder:

- **`leaderboard.csv`**: An aggregate ranking of every single hyperparameter combination tested, sorted by which model performed best on the blind validation data.
- **`run_opt_<optimizer>_lr_<lr>_l2_<l2>.csv`**: An epoch-by-epoch trajectory of a specific run. You can open this in Excel to graph exactly how the loss dropped and how the weights shifted over time.
- **`run_opt_<optimizer>_lr_<lr>_l2_<l2>.json`**: A highly detailed JSON dump containing summary statistics, final equations, and error distributions.

### Key Metrics to Watch
* **Validation Loss vs Train Loss**: If Train Loss goes down but Validation Loss goes up, your model is *overfitting* (memorizing the answers instead of learning the concepts).
* **Generalization Gap**: The mathematical difference between Train and Validation loss.
* **Gradient Norm**: Think of the gradient as the "slope of the error hill". If the gradient norm is near `0`, the model has reached the bottom of the valley and is fully converged. If it's high, the model is still rolling down the hill.
