using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MyMLLab;

/// <summary>
/// Application entrypoint for MyMLLab.
/// Orchestrates dataset preparation, experiment sweeps, training, ranking, and artifact output.
/// </summary>
internal static class Program
{
    /// <summary>
    /// Runs the full ML experiment pipeline from configuration through ranked reporting.
    /// </summary>
    private static void Main()
    {
        // Global dataset configuration: controls synthetic data shape, noise, and reproducibility.
        // We use a predefined configuration to generate the exact same "fake" dataset every time.
        var datasetConfig = new DatasetDefinition(
            Count: 240,             // Total number of data points to generate.
            Slope: 2.5,             // The "true" weight (w) the model needs to learn (y = 2.5x - 0.8).
            Intercept: -0.8,        // The "true" bias (b) the model needs to learn.
            MinX: -2.0,             // The lower bound for our input feature 'X'.
            MaxX: 2.0,              // The upper bound for our input feature 'X'.
            NoiseStdDev: 0.25,      // The amount of randomness/noise added to 'Y' to make it realistic.
            Seed: 42,               // Fixed random seed ensures we get the exact same numbers every run.
            TrainRatio: 0.8,        // 80% of data used for training, 20% held back for validation.
            NormalizeFeature: true  // Whether to scale X features to have a mean of 0 and standard dev of 1.
        );

        // Data pipeline: synthesize examples, split train/validation, optionally normalize features.
        // 'split' contains both the Training array and Validation array we will use.
        var split = DatasetEngine.Prepare(datasetConfig);

        // Experiment configuration: defines the sweep space (hyperparameter tuning) and training controls.
        // We want to test different combinations to see which learns the "true" slope and intercept best.
        var experiment = new ExperimentDefinition(
            Epochs: 220,                                     // Maximum number of full passes through the training data.
            EarlyStoppingPatience: 25,                       // Stop early if validation loss doesn't improve for 25 epochs.
            InitialWeight: 0.0,                              // Starting guess for the slope.
            InitialBias: 0.0,                                // Starting guess for the intercept.
            LearningRates: new[] { 0.001, 0.005, 0.01, 0.03, 0.05, 0.1 }, // Step sizes for weight updates.
            Optimizers: new[] { OptimizerKind.Sgd, OptimizerKind.Adam },  // The algorithms used to update weights.
            L2Penalties: new[] { 0.0, 0.001, 0.01 },         // Regularization to penalize huge weights and prevent overfitting.
            OutputDirectory: "artifacts"                     // Folder name where CSV and JSON logs will be saved.
        );

        // Execute all runs (every combination of optimizer, learning rate, and L2 penalty).
        // Then sort the array so the runs with the lowest validation loss (lowest error on unseen data) appear first.
        var results = ExperimentRunner.Run(split, experiment).OrderBy(r => r.FinalValidationLoss).ToArray();

        // Keep console output concise while still preserving full details in artifacts.
        // 'maxRows' limits our printed leaderboard to either 12 rows or the total number of runs, whichever is smaller.
        var maxRows = Math.Min(12, results.Length);

        // Print leaderboard header.
        Console.WriteLine($"MyMLLab experiment leaderboard (showing top {maxRows} of {results.Length})");
        Console.WriteLine(new string('-', 140));
        Console.WriteLine("rank opt  lr     l2      train_loss  val_loss    gap         grad_norm  epochs  status        w        b");

        // Print top-N leaderboard rows using a loop.
        for (var i = 0; i < maxRows; i++)
        {
            // 'result' holds the summary of one specific combination of hyperparameters.
            var result = results[i];
            Console.WriteLine(
                $"{i + 1,4} {result.Optimizer,-4} {result.LearningRate,6:0.###} {result.L2Penalty,7:0.###} " +
                $"{result.FinalTrainingLoss,10:0.000000} {result.FinalValidationLoss,10:0.000000} {result.GeneralizationGap,10:0.000000} " +
                $"{result.FinalGradientNorm,10:0.000000} {result.EpochsRan,6}  {result.ConvergenceStatus,-12} {result.FinalWeight,8:0.0000} {result.FinalBias,8:0.0000}");
        }

        // Surface where CSV/JSON artifacts were saved so the user knows where to find the full data.
        Console.WriteLine($"\nArtifacts written to: {Path.GetFullPath(experiment.OutputDirectory)}");
    }
}

/// <summary>
/// Responsible for dataset generation, splitting, and optional feature normalization.
/// </summary>
internal static class DatasetEngine
{
    /// <summary>
    /// Builds a complete dataset split from a dataset definition.
    /// </summary>
    /// <param name="definition">Synthetic-data configuration and preprocessing toggles.</param>
    /// <returns>Train/validation split and normalization metadata.</returns>
    public static DatasetSplit Prepare(DatasetDefinition definition)
    {
        // Generate raw synthetic points. 'raw' is a list of X,Y pairs generated from our underlying linear equation.
        var raw = GenerateLinear(definition);

        // Compute deterministic split boundary. 'trainCount' calculates how many points belong in the training set.
        var trainCount = (int)(raw.Count * definition.TrainRatio);

        // 'train' contains the data the model will look at to adjust its weights.
        var train = raw.Take(trainCount).ToArray();
        // 'validation' contains the data the model is completely blind to during training; used to measure true accuracy.
        var validation = raw.Skip(trainCount).ToArray();

        // If normalization is disabled, return data as-is.
        if (!definition.NormalizeFeature)
        {
            return new DatasetSplit(train, validation, Normalization.None);
        }

        // Compute normalization stats from train only to avoid validation leakage (peeking at validation data).
        // 'mean' is the average of all X values in the training set.
        var mean = train.Average(p => p.X);

        // 'variance' measures how spread out the X values are from the mean.
        var variance = train.Average(p => (p.X - mean) * (p.X - mean));

        // 'stdDev' is the square root of variance, giving us the standard deviation. We use Max to avoid division by zero.
        var stdDev = Math.Sqrt(Math.Max(variance, 1e-12));

        // Apply same train-derived normalization to both partitions. This centers the data around 0, making gradient descent faster.
        // 'normalizedTrain' adjusts every training point based on the mean and standard deviation.
        var normalizedTrain = train.Select(p => new DataPoint((p.X - mean) / stdDev, p.Y)).ToArray();

        // 'normalizedValidation' scales validation points using the TRAINING mean and stdDev.
        var normalizedValidation = validation.Select(p => new DataPoint((p.X - mean) / stdDev, p.Y)).ToArray();

        return new DatasetSplit(normalizedTrain, normalizedValidation, new Normalization(mean, stdDev));
    }

    /// <summary>
    /// Creates synthetic linear-regression data with additive Gaussian noise.
    /// </summary>
    /// <param name="definition">Parameters like count, bounds, slope, and intercept.</param>
    private static IReadOnlyList<DataPoint> GenerateLinear(DatasetDefinition definition)
    {
        // Seeded RNG keeps experiments reproducible. 'random' is our random number generator.
        var random = new Random(definition.Seed);

        // 'points' is the list we will populate with our generated data points.
        var points = new List<DataPoint>(capacity: definition.Count);

        // Build all points one by one. 'i' is the current loop index.
        for (var i = 0; i < definition.Count; i++)
        {
            // Sample 'x' uniformly from configured range (between MinX and MaxX).
            var x = definition.MinX + (definition.MaxX - definition.MinX) * random.NextDouble();

            // Sample additive Gaussian noise. 'noise' acts as real-world variance (e.g. sensor errors).
            var noise = NextGaussian(random) * definition.NoiseStdDev;

            // Generate target according to y = slope*x + intercept + noise. 'y' is the expected output.
            var y = definition.Slope * x + definition.Intercept + noise;

            points.Add(new DataPoint(x, y));
        }

        return points;
    }

    /// <summary>
    /// Generates a standard-normal random number using the Box-Muller transform.
    /// Box-Muller transforms uniformly distributed random variables into normal (bell-curve) distribution.
    /// </summary>
    /// <param name="random">The random number generator instance.</param>
    private static double NextGaussian(Random random)
    {
        // 'u1' and 'u2' are two independent uniformly distributed random numbers between 0 and 1.
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();

        // Math magic that converts flat randomness into bell-curve randomness.
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}

/// <summary>
/// Executes the experiment sweep over optimizers, learning rates, and L2 penalties.
/// </summary>
internal static class ExperimentRunner
{
    /// <summary>
    /// Runs all configured experiment combinations and returns per-run summaries.
    /// </summary>
    /// <param name="split">The prepared Train/Validation dataset split.</param>
    /// <param name="definition">The hyperparameters mapping to run combinations for.</param>
    public static IReadOnlyList<ExperimentResult> Run(DatasetSplit split, ExperimentDefinition definition)
    {
        // Ensure output location exists before writing artifacts.
        Directory.CreateDirectory(definition.OutputDirectory);

        // 'results' will store the final performance summary of every single combination tested.
        var results = new List<ExperimentResult>();

        // Cartesian product over all configured knobs: Tests every combination of Optimizer, LR, and L2.
        foreach (var optimizer in definition.Optimizers) // 'optimizer' is either Sgd or Adam
        {
            foreach (var learningRate in definition.LearningRates) // 'learningRate' is the current step size
            {
                foreach (var l2 in definition.L2Penalties) // 'l2' is the current regularization penalty
                {
                    // New model instance per run for isolation. 'model' starts fresh with 0 slope and 0 intercept.
                    var model = new LinearRegressionModel(definition.InitialWeight, definition.InitialBias);

                    // Train once for this hyperparameter combination. 'history' tracks the metrics over all epochs.
                    var history = TrainingEngine.Train(
                        model,
                        split,
                        new TrainingConfig(
                            Epochs: definition.Epochs,
                            LearningRate: learningRate,
                            Optimizer: optimizer,
                            L2Penalty: l2,
                            EarlyStoppingPatience: definition.EarlyStoppingPatience));

                    // Compute error-distribution stats on validation set. 
                    // 'finalErrors' is an array of raw differences between what the model predicted and the true Y.
                    var finalErrors = split.Validation.Select(p => model.Predict(p.X) - p.Y).ToArray();

                    // 'errorStats' aggregates those raw errors into Mean, StdDev, and percentiles.
                    var errorStats = ErrorStatistics.From(finalErrors);

                    // Pull latest tracked metrics for summary.
                    // 'finalTrainLoss' is the mean squared error on training data at the last epoch.
                    var finalTrainLoss = history.TrainingLossByEpoch[^1];
                    // 'finalValLoss' is the mean squared error on validation data at the last epoch.
                    var finalValLoss = history.ValidationLossByEpoch[^1];
                    // 'finalGradNorm' tells us how close the model was to a flat minimum (0 means completely flat/converged).
                    var finalGradNorm = history.GradientNormByEpoch[^1];
                    // 'epochsRan' is how many passes occurred before early stopping triggered (or max epochs hit).
                    var epochsRan = history.TrainingLossByEpoch.Count;
                    // 'gap' measures overfitting. A high gap means the model memorized the training data but fails on validation data.
                    var gap = finalValLoss - finalTrainLoss;

                    // Build compact summary row for this run. 'summary' holds all this data.
                    var summary = new ExperimentResult(
                        Optimizer: optimizer,
                        LearningRate: learningRate,
                        L2Penalty: l2,
                        FinalTrainingLoss: finalTrainLoss,
                        FinalValidationLoss: finalValLoss,
                        FinalWeight: history.WeightByEpoch[^1],
                        FinalBias: history.BiasByEpoch[^1],
                        BestEpoch: history.BestEpoch,
                        EpochsRan: epochsRan,
                        FinalGradientNorm: finalGradNorm,
                        GeneralizationGap: gap,
                        ConvergenceStatus: RunDiagnostics.AssessConvergence(finalGradNorm, gap, history.BestEpoch, epochsRan),
                        ErrorMean: errorStats.Mean,
                        ErrorStdDev: errorStats.StdDev,
                        ErrorP50: errorStats.P50,
                        ErrorP90: errorStats.P90);

                    results.Add(summary);

                    // Write per-run artifacts (CSV/JSON) immediately to disk in case the program crashes halfway.
                    MetricsLogger.Write(definition.OutputDirectory, summary, history, split.Normalization);
                }
            }
        }

        // Write aggregate leaderboard (summary of all combinations) to disk.
        MetricsLogger.WriteLeaderboard(definition.OutputDirectory, results);
        return results;
    }
}

/// <summary>
/// Handles model parameter updates and per-epoch metric tracking.
/// </summary>
internal static class TrainingEngine
{
    /// <summary>
    /// Trains a linear model using the requested optimizer and settings.
    /// </summary>
    /// <param name="model">The linear model containing weights to update.</param>
    /// <param name="split">Train and Validation datasets.</param>
    /// <param name="config">Training configurations (like epochs, learning rate).</param>
    /// <returns>A full epoch-by-epoch history of metrics.</returns>
    public static TrainingHistory Train(LinearRegressionModel model, DatasetSplit split, TrainingConfig config)
    {
        // Allocate metric history buffers. These lists store performance data for every single epoch.
        var trainingLoss = new List<double>(config.Epochs);     // 'trainingLoss' tracks MSE on training set over time.
        var validationLoss = new List<double>(config.Epochs);   // 'validationLoss' tracks MSE on validation set over time.
        var weights = new List<double>(config.Epochs);          // 'weights' tracks the learned slope value over time.
        var biases = new List<double>(config.Epochs);           // 'biases' tracks the learned intercept value over time.
        var gradNorms = new List<double>(config.Epochs);        // 'gradNorms' tracks the size of the update vector.

        // Best-checkpoint tracking for early stopping restore.
        // If the model starts getting worse (overfitting), we want to rewind to the "best" historical state.
        var bestValidation = double.MaxValue; // 'bestValidation' tracks the lowest error seen so far. Starts at infinity.
        var bestEpoch = 0;                    // 'bestEpoch' tracks which epoch achieved the lowest error.
        var bestWeight = model.Weight;        // 'bestWeight' saves the slope from that best epoch.
        var bestBias = model.Bias;            // 'bestBias' saves the intercept from that best epoch.
        var stagnantEpochs = 0;               // 'stagnantEpochs' counts how many epochs in a row have failed to beat bestValidation.

        // Adam optimizer moment/variance state.
        // Adam relies on past gradients to adaptively tune the learning rate for each parameter.
        var mW = 0.0;       // 'mW' is the first moment (running average of past gradients) for the Weight.
        var mB = 0.0;       // 'mB' is the first moment (running average of past gradients) for the Bias.
        var vW = 0.0;       // 'vW' is the second moment (running average of squared past gradients) for the Weight.
        var vB = 0.0;       // 'vB' is the second moment (running average of squared past gradients) for the Bias.

        // Standard Adam hyperparameters (often left to default values in ML literature).
        const double beta1 = 0.9;     // 'beta1' controls how fast the first moment decays (memory of past gradients).
        const double beta2 = 0.999;   // 'beta2' controls how fast the second moment decays (memory of past squared gradients).
        const double epsilon = 1e-8;  // 'epsilon' is a tiny number added to division steps to prevent dividing by exactly zero.

        // Main epoch loop. 'epoch' represents one complete pass through the entire training dataset.
        for (var epoch = 1; epoch <= config.Epochs; epoch++)
        {
            // Accumulate full-batch gradients. 
            // A gradient tells us the "direction" and "steepness" of the error. We want to walk downhill.
            var gradW = 0.0; // 'gradW' will sum up the gradient for the slope.
            var gradB = 0.0; // 'gradB' will sum up the gradient for the intercept.

            // Iterate through every single point in the training set to calculate the total error (Full-Batch Gradient Descent).
            foreach (var point in split.Train)
            {
                // 'prediction' is our model's current guess.
                var prediction = model.Predict(point.X);
                // 'error' is the raw difference between the guess and the truth.
                var error = prediction - point.Y;

                // Accumulate partial derivatives.
                // The derivative of MSE with respect to weight involves multiplying the error by X.
                gradW += error * point.X;
                // The derivative of MSE with respect to bias is just the error itself.
                gradB += error;
            }

            // Convert sums to Mean Squared Error (MSE) gradients.
            // 'trainSize' is the number of points. We divide by this to get the average gradient, multiplying by 2 per MSE derivative math.
            var trainSize = split.Train.Length;
            gradW = (2.0 / trainSize) * gradW;
            gradB = (2.0 / trainSize) * gradB;

            // Add L2 term for weight only. (L2 Regularization / Ridge Regression)
            // L2 artificially increases the gradient based on how large the weight is, forcing the model to prefer smaller weights.
            gradW += config.L2Penalty * model.Weight;

            // Branch on optimizer to update parameters based on gradients.
            if (config.Optimizer == OptimizerKind.Adam)
            {
                // Update first moments (momentum). 
                mW = beta1 * mW + (1.0 - beta1) * gradW;
                mB = beta1 * mB + (1.0 - beta1) * gradB;

                // Update second moments (uncentered variance / RMS).
                vW = beta2 * vW + (1.0 - beta2) * gradW * gradW;
                vB = beta2 * vB + (1.0 - beta2) * gradB * gradB;

                // Bias-corrected moments. 
                // At early epochs, moments are biased towards 0. This step warms them up artificially.
                // 'mWHat', 'mBHat', 'vWHat', 'vBHat' are the corrected versions.
                var mWHat = mW / (1.0 - Math.Pow(beta1, epoch));
                var mBHat = mB / (1.0 - Math.Pow(beta1, epoch));
                var vWHat = vW / (1.0 - Math.Pow(beta2, epoch));
                var vBHat = vB / (1.0 - Math.Pow(beta2, epoch));

                // Parameter updates using Adam formula.
                // Weights update dynamically based on learning rate, scaled by momentum over standard deviation.
                model.Weight -= config.LearningRate * mWHat / (Math.Sqrt(vWHat) + epsilon);
                model.Bias -= config.LearningRate * mBHat / (Math.Sqrt(vBHat) + epsilon);
            }
            else
            {
                // Vanilla SGD (Stochastic Gradient Descent, though technically Full-Batch here).
                // Multiply gradient by fixed learning rate, and subtract it to move "downhill" in error space.
                model.Weight -= config.LearningRate * gradW;
                model.Bias -= config.LearningRate * gradB;
            }

            // Evaluate losses after update.
            // 'trainLoss' shows how well we fit the training data.
            var trainLoss = CalculateMse(model, split.Train);
            // 'valLoss' shows how well we fit the unseen validation data.
            var valLoss = CalculateMse(model, split.Validation);

            // Persist per-epoch metrics to arrays.
            trainingLoss.Add(trainLoss);
            validationLoss.Add(valLoss);
            weights.Add(model.Weight);
            biases.Add(model.Bias);

            // 'gradNorms.Add' computes the Euclidean length (magnitude) of the gradient vector.
            // If it approaches 0, the model has found the bottom of the error curve (converged).
            gradNorms.Add(Math.Sqrt(gradW * gradW + gradB * gradB));

            // Best-checkpoint update and early-stop bookkeeping.
            // We only care about validation loss. If it drops, we have a new "best" model.
            if (valLoss < bestValidation)
            {
                bestValidation = valLoss;
                bestEpoch = epoch;
                bestWeight = model.Weight;
                bestBias = model.Bias;
                stagnantEpochs = 0; // Reset patience counter since we improved.
            }
            else
            {
                stagnantEpochs++; // Increment patience counter since the model didn't improve.
            }

            // Early stopping condition. If we haven't improved in X epochs, abort training to save time and prevent overfitting.
            if (stagnantEpochs >= config.EarlyStoppingPatience)
            {
                break;
            }
        }

        // Restore best validation checkpoint.
        // We rewind the model to its absolute best state, throwing away any overfit parameters from later epochs.
        model.Weight = bestWeight;
        model.Bias = bestBias;

        return new TrainingHistory(trainingLoss, validationLoss, weights, biases, gradNorms, bestEpoch);
    }

    /// <summary>
    /// Computes mean squared error (MSE) for a model over a given dataset.
    /// MSE squares the errors so large mistakes are punished disproportionately.
    /// </summary>
    /// <param name="model">The model to use for predictions.</param>
    /// <param name="points">The dataset to calculate the error against.</param>
    private static double CalculateMse(LinearRegressionModel model, IReadOnlyList<DataPoint> points)
    {
        // 'squaredError' acts as a running total.
        var squaredError = 0.0;

        foreach (var point in points) // 'point' is the current data pair
        {
            // 'error' is the distance between prediction and truth.
            var error = model.Predict(point.X) - point.Y;

            // Squaring it prevents negative and positive errors from canceling each other out.
            squaredError += error * error;
        }

        // Divide by the total count to get the "Mean" (average).
        return squaredError / points.Count;
    }
}

/// <summary>
/// Writes per-run and aggregate experiment artifacts to disk.
/// </summary>
internal static class MetricsLogger
{
    /// <summary>
    /// Writes one run's metrics in CSV and JSON formats.
    /// </summary>
    /// <param name="outputDirectory">Folder to write to.</param>
    /// <param name="summary">The high-level result statistics.</param>
    /// <param name="history">The epoch-by-epoch historical metrics.</param>
    /// <param name="normalization">Data indicating if and how scaling was applied.</param>
    public static void Write(string outputDirectory, ExperimentResult summary, TrainingHistory history, Normalization normalization)
    {
        // Build deterministic run id used in file names based on hyperparameters.
        // 'runId' ensures each CSV/JSON has a unique file name.
        var runId = BuildRunId(summary.Optimizer, summary.LearningRate, summary.L2Penalty);

        // 'csvPath' and 'jsonPath' represent the full file paths.
        var csvPath = Path.Combine(outputDirectory, $"run_{runId}.csv");
        var jsonPath = Path.Combine(outputDirectory, $"run_{runId}.json");

        // Build CSV payload (epoch-by-epoch trajectory).
        // 'csv' uses StringBuilder for memory-efficient string concatenation.
        var csv = new StringBuilder();
        csv.AppendLine("epoch,training_loss,validation_loss,weight,bias,gradient_norm");

        // Iterate through all epochs to write out the row data.
        for (var i = 0; i < history.TrainingLossByEpoch.Count; i++)
        {
            csv.AppendLine(string.Join(',',
                i + 1,
                history.TrainingLossByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.ValidationLossByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.WeightByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.BiasByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.GradientNormByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(csvPath, csv.ToString());

        // Build rich JSON payload for programmatic inspection (e.g. visualizing in Python later).
        // 'payload' is an anonymous object that captures literally everything about the run.
        var payload = new
        {
            summary.Optimizer,
            summary.LearningRate,
            summary.L2Penalty,
            summary.FinalTrainingLoss,
            summary.FinalValidationLoss,
            summary.FinalWeight,
            summary.FinalBias,
            summary.BestEpoch,
            summary.EpochsRan,
            summary.FinalGradientNorm,
            summary.GeneralizationGap,
            summary.ConvergenceStatus,
            summary.ErrorMean,
            summary.ErrorStdDev,
            summary.ErrorP50,
            summary.ErrorP90,
            normalization.Mean,
            normalization.StdDev,
            history.TrainingLossByEpoch,
            history.ValidationLossByEpoch,
            history.WeightByEpoch,
            history.BiasByEpoch,
            history.GradientNormByEpoch
        };

        File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>
    /// Writes aggregate leaderboard CSV sorted by final validation loss.
    /// </summary>
    /// <param name="outputDirectory">Folder to output to.</param>
    /// <param name="results">List of all completed experiments.</param>
    public static void WriteLeaderboard(string outputDirectory, IEnumerable<ExperimentResult> results)
    {
        // 'ordered' sorts the results from best (lowest loss) to worst.
        var ordered = results.OrderBy(r => r.FinalValidationLoss).ToArray();
        // 'path' is the file location for the aggregate leaderboard.
        var path = Path.Combine(outputDirectory, "leaderboard.csv");

        // 'csv' holds the text string of the file.
        var csv = new StringBuilder();
        csv.AppendLine("rank,optimizer,learning_rate,l2_penalty,final_training_loss,final_validation_loss,generalization_gap,final_gradient_norm,best_epoch,epochs_ran,convergence_status,final_weight,final_bias,error_mean,error_std_dev,error_p50,error_p90");

        // 'i' represents the rank (0-indexed).
        for (var i = 0; i < ordered.Length; i++)
        {
            // 'row' is the specific experiment result we are printing.
            var row = ordered[i];
            csv.AppendLine(string.Join(',',
                i + 1,
                row.Optimizer,
                row.LearningRate.ToString("0.###", CultureInfo.InvariantCulture),
                row.L2Penalty.ToString("0.###", CultureInfo.InvariantCulture),
                row.FinalTrainingLoss.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalValidationLoss.ToString("0.000000", CultureInfo.InvariantCulture),
                row.GeneralizationGap.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalGradientNorm.ToString("0.000000", CultureInfo.InvariantCulture),
                row.BestEpoch,
                row.EpochsRan,
                row.ConvergenceStatus,
                row.FinalWeight.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalBias.ToString("0.000000", CultureInfo.InvariantCulture),
                row.ErrorMean.ToString("0.000000", CultureInfo.InvariantCulture),
                row.ErrorStdDev.ToString("0.000000", CultureInfo.InvariantCulture),
                row.ErrorP50.ToString("0.000000", CultureInfo.InvariantCulture),
                row.ErrorP90.ToString("0.000000", CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(path, csv.ToString());
    }

    /// <summary>
    /// Builds a compact, deterministic slug used for run artifact filenames.
    /// </summary>
    /// <param name="optimizer">Algorithm type.</param>
    /// <param name="learningRate">Step size.</param>
    /// <param name="l2">Penalty value.</param>
    private static string BuildRunId(OptimizerKind optimizer, double learningRate, double l2)
    {
        // Format the numbers into safe strings for file names.
        // 'lrSlug' and 'l2Slug' replace decimal points with underscores.
        var lrSlug = learningRate.ToString("0.###", CultureInfo.InvariantCulture).Replace('.', '_');
        var l2Slug = l2.ToString("0.###", CultureInfo.InvariantCulture).Replace('.', '_');
        return $"opt_{optimizer}_lr_{lrSlug}_l2_{l2Slug}".ToLowerInvariant();
    }
}

/// <summary>
/// Produces simple human-readable convergence labels from run-level metrics.
/// </summary>
internal static class RunDiagnostics
{
    /// <summary>
    /// Assigns a convergence status string using heuristic thresholds to diagnose common ML problems.
    /// </summary>
    /// <param name="gradientNorm">Size of gradient at finish. Large means still moving.</param>
    /// <param name="generalizationGap">Validation minus Train loss. Large means overfitting.</param>
    /// <param name="bestEpoch">When the model peaked.</param>
    /// <param name="epochsRan">How long it trained for.</param>
    public static string AssessConvergence(double gradientNorm, double generalizationGap, int bestEpoch, int epochsRan)
    {
        // Underfit: Gradient is still huge but we ran for many epochs, indicating it's stuck or learning too slowly.
        if (gradientNorm > 0.5 && epochsRan >= 200)
        {
            return "underfit";
        }

        // Still-learning: Gradient is active, meaning the model hasn't found the bottom of the error curve yet.
        if (gradientNorm > 0.1)
        {
            return "still-learning";
        }

        // Overfit: The gap between Train and Validation is large, meaning the model memorized the training data.
        if (generalizationGap > 0.03)
        {
            return "overfit";
        }

        // Plateau: The best epoch was a long time ago, meaning it flattened out and just bounced around uselessly at the end.
        if (bestEpoch < epochsRan - 15)
        {
            return "plateau";
        }

        // Stable: None of the bad things happened. The model likely found a good minimum.
        return "stable";
    }
}

/// <summary>
/// Computes descriptive statistics for prediction-error arrays.
/// </summary>
internal static class ErrorStatistics
{
    /// <summary>
    /// Computes mean, standard deviation, and selected quantiles (percentiles).
    /// </summary>
    /// <param name="errors">Array of differences between predictions and reality.</param>
    public static ErrorSummary From(IReadOnlyList<double> errors)
    {
        // 'ordered' sorts errors smallest to largest to compute percentiles.
        var ordered = errors.OrderBy(e => e).ToArray();

        // 'mean' is the average error.
        var mean = ordered.Average();
        // 'variance' measures spread of the errors.
        var variance = ordered.Average(e => (e - mean) * (e - mean));

        return new ErrorSummary(
            Mean: mean,
            StdDev: Math.Sqrt(variance), // Standard deviation is square root of variance.
            P50: Quantile(ordered, 0.50), // 50th percentile (Median).
            P90: Quantile(ordered, 0.90)  // 90th percentile (How bad are the worst 10% of errors?).
        );
    }

    /// <summary>
    /// Computes a linearly-interpolated quantile from a sorted array.
    /// </summary>
    /// <param name="ordered">The sorted array of error values.</param>
    /// <param name="q">The percentile decimal (e.g. 0.50 for Median).</param>
    private static double Quantile(IReadOnlyList<double> ordered, double q)
    {
        if (ordered.Count == 0)
        {
            return 0.0;
        }

        // 'index' calculates the theoretical array index for this percentile.
        var index = (ordered.Count - 1) * q;

        // 'lower' is the index rounded down.
        var lower = (int)Math.Floor(index);
        // 'upper' is the index rounded up.
        var upper = (int)Math.Ceiling(index);

        // If it lands perfectly on an integer index, just return that value.
        if (lower == upper)
        {
            return ordered[lower];
        }

        // 'weight' determines how close the theoretical index is to the upper vs lower boundary to interpolate gracefully.
        var weight = index - lower;
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
    }
}

/// <summary>
/// Minimal one-feature linear model: y_hat = w*x + b.
/// </summary>
/// <param name="initialWeight">Starting point for the slope parameter.</param>
/// <param name="initialBias">Starting point for the intercept parameter.</param>
internal sealed class LinearRegressionModel(double initialWeight, double initialBias)
{
    /// <summary>
    /// Trainable slope parameter. Multiplied against the X input.
    /// </summary>
    public double Weight { get; set; } = initialWeight;

    /// <summary>
    /// Trainable intercept parameter. Added unconditionally as a baseline offset.
    /// </summary>
    public double Bias { get; set; } = initialBias;

    /// <summary>
    /// Predicts target value for one scalar feature input.
    /// </summary>
    /// <param name="x">The input feature value.</param>
    public double Predict(double x) => Weight * x + Bias;
}

/// <summary>
/// Supported optimizer algorithms for this lab.
/// </summary>
internal enum OptimizerKind
{
    /// <summary>
    /// Vanilla stochastic gradient descent (full-batch in current implementation).
    /// Uses a raw, static learning rate against the gradient.
    /// </summary>
    Sgd,

    /// <summary>
    /// Adaptive Moment Estimation optimizer.
    /// Highly robust to different scales; maintains individual learning rates per parameter based on past momentum.
    /// </summary>
    Adam
}

/// <summary>
/// One supervised sample pair (X feature, Y target).
/// </summary>
/// <param name="X">The input feature data.</param>
/// <param name="Y">The expected target/ground truth.</param>
internal readonly record struct DataPoint(double X, double Y);

/// <summary>
/// Feature normalization metadata used when normalization is enabled.
/// </summary>
/// <param name="Mean">The average of the training data X values.</param>
/// <param name="StdDev">The standard deviation of the training data X values.</param>
internal readonly record struct Normalization(double Mean, double StdDev)
{
    /// <summary>
    /// Sentinel values used when normalization is disabled (leaves data completely unchanged when applied).
    /// </summary>
    public static Normalization None => new(0.0, 1.0);
}

/// <summary>
/// Configuration describing synthetic dataset generation and preprocessing.
/// </summary>
/// <param name="Count">Number of rows to generate.</param>
/// <param name="Slope">True slope (for the fake data logic).</param>
/// <param name="Intercept">True intercept (for the fake data logic).</param>
/// <param name="MinX">Lower bound of generation.</param>
/// <param name="MaxX">Upper bound of generation.</param>
/// <param name="NoiseStdDev">Randomness multiplier applied to Y.</param>
/// <param name="Seed">Number used to fix randomness for reproducibility.</param>
/// <param name="TrainRatio">Percent (0.0 to 1.0) of data allocated to training vs validation.</param>
/// <param name="NormalizeFeature">Boolean toggle to center data scale to mean 0, variance 1.</param>
internal sealed record DatasetDefinition(
    int Count,
    double Slope,
    double Intercept,
    double MinX,
    double MaxX,
    double NoiseStdDev,
    int Seed,
    double TrainRatio,
    bool NormalizeFeature);

/// <summary>
/// Prepared train/validation partitions plus normalization metadata.
/// </summary>
/// <param name="Train">Data array used exclusively for training.</param>
/// <param name="Validation">Data array used exclusively for testing unseen accuracy.</param>
/// <param name="Normalization">Information about how the dataset was scaled.</param>
internal sealed record DatasetSplit(DataPoint[] Train, DataPoint[] Validation, Normalization Normalization);

/// <summary>
/// Experiment sweep definition across training hyperparameters.
/// </summary>
/// <param name="Epochs">Max total sweeps through data.</param>
/// <param name="EarlyStoppingPatience">Number of non-improving epochs before abandoning train.</param>
/// <param name="InitialWeight">Starting guess for w.</param>
/// <param name="InitialBias">Starting guess for b.</param>
/// <param name="LearningRates">Array of step sizes to try.</param>
/// <param name="Optimizers">Array of optimizer algorithms to try.</param>
/// <param name="L2Penalties">Array of ridge regularization strengths to try.</param>
/// <param name="OutputDirectory">Folder for metric artifact files.</param>
internal sealed record ExperimentDefinition(
    int Epochs,
    int EarlyStoppingPatience,
    double InitialWeight,
    double InitialBias,
    double[] LearningRates,
    OptimizerKind[] Optimizers,
    double[] L2Penalties,
    string OutputDirectory);

/// <summary>
/// One concrete training run configuration.
/// </summary>
internal sealed record TrainingConfig(
    int Epochs,
    double LearningRate,
    OptimizerKind Optimizer,
    double L2Penalty,
    int EarlyStoppingPatience);

/// <summary>
/// Compact summary for one completed run in the experiment sweep.
/// </summary>
internal sealed record ExperimentResult(
    OptimizerKind Optimizer,
    double LearningRate,
    double L2Penalty,
    double FinalTrainingLoss,
    double FinalValidationLoss,
    double FinalWeight,
    double FinalBias,
    int BestEpoch,
    int EpochsRan,
    double FinalGradientNorm,
    double GeneralizationGap,
    string ConvergenceStatus,
    double ErrorMean,
    double ErrorStdDev,
    double ErrorP50,
    double ErrorP90);

/// <summary>
/// Descriptive statistics computed from prediction errors.
/// </summary>
internal sealed record ErrorSummary(double Mean, double StdDev, double P50, double P90);

/// <summary>
/// Full per-epoch history captured during training.
/// </summary>
internal sealed record TrainingHistory(
    IReadOnlyList<double> TrainingLossByEpoch,
    IReadOnlyList<double> ValidationLossByEpoch,
    IReadOnlyList<double> WeightByEpoch,
    IReadOnlyList<double> BiasByEpoch,
    IReadOnlyList<double> GradientNormByEpoch,
    int BestEpoch);
