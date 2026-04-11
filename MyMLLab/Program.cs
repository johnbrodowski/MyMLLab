using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MyMLLab;

internal static class Program
{
    private static void Main()
    {
        // Keep all high-level knobs visible in one place for easy experimentation.
        var datasetConfig = new DatasetDefinition(
            Count: 240,
            Slope: 2.5,
            Intercept: -0.8,
            MinX: -2.0,
            MaxX: 2.0,
            NoiseStdDev: 0.25,
            Seed: 42,
            TrainRatio: 0.8,
            NormalizeFeature: true);

        // Data pipeline: synthesize -> split -> (optional) normalize.
        var split = DatasetEngine.Prepare(datasetConfig);

        // Broader sweep gives better coverage of optimizer hyperparameter interactions.
        var experiment = new ExperimentDefinition(
            Epochs: 220,
            EarlyStoppingPatience: 25,
            InitialWeight: 0.0,
            InitialBias: 0.0,
            LearningRates: new[] { 0.001, 0.005, 0.01, 0.03, 0.05, 0.1 },
            Optimizers: new[] { OptimizerKind.Sgd, OptimizerKind.Adam },
            L2Penalties: new[] { 0.0, 0.001, 0.01 },
            OutputDirectory: "artifacts");

        var results = ExperimentRunner.Run(split, experiment).OrderBy(r => r.FinalValidationLoss).ToArray();

        Console.WriteLine("MyMLLab experiment leaderboard (best validation loss first)");
        Console.WriteLine(new string('-', 140));
        Console.WriteLine("rank opt  lr     l2      train_loss  val_loss    gap         grad_norm  epochs  status        w        b");

        for (var i = 0; i < results.Length; i++)
        {
            var result = results[i];
            Console.WriteLine(
                $"{i + 1,4} {result.Optimizer,-4} {result.LearningRate,6:0.###} {result.L2Penalty,7:0.###} " +
                $"{result.FinalTrainingLoss,10:0.000000} {result.FinalValidationLoss,10:0.000000} {result.GeneralizationGap,10:0.000000} " +
                $"{result.FinalGradientNorm,10:0.000000} {result.EpochsRan,6}  {result.ConvergenceStatus,-12} {result.FinalWeight,8:0.0000} {result.FinalBias,8:0.0000}");
        }

        Console.WriteLine($"\nArtifacts written to: {Path.GetFullPath(experiment.OutputDirectory)}");
    }
}

internal static class DatasetEngine
{
    public static DatasetSplit Prepare(DatasetDefinition definition)
    {
        var raw = GenerateLinear(definition);
        var trainCount = (int)(raw.Count * definition.TrainRatio);
        var train = raw.Take(trainCount).ToArray();
        var validation = raw.Skip(trainCount).ToArray();

        // Normalize feature using only train partition statistics to avoid data leakage.
        if (!definition.NormalizeFeature)
        {
            return new DatasetSplit(train, validation, Normalization.None);
        }

        var mean = train.Average(p => p.X);
        var variance = train.Average(p => (p.X - mean) * (p.X - mean));
        var stdDev = Math.Sqrt(Math.Max(variance, 1e-12));

        var normalizedTrain = train.Select(p => new DataPoint((p.X - mean) / stdDev, p.Y)).ToArray();
        var normalizedValidation = validation.Select(p => new DataPoint((p.X - mean) / stdDev, p.Y)).ToArray();

        return new DatasetSplit(normalizedTrain, normalizedValidation, new Normalization(mean, stdDev));
    }

    private static IReadOnlyList<DataPoint> GenerateLinear(DatasetDefinition definition)
    {
        var random = new Random(definition.Seed);
        var points = new List<DataPoint>(capacity: definition.Count);

        for (var i = 0; i < definition.Count; i++)
        {
            var x = definition.MinX + (definition.MaxX - definition.MinX) * random.NextDouble();
            var noise = NextGaussian(random) * definition.NoiseStdDev;
            var y = definition.Slope * x + definition.Intercept + noise;
            points.Add(new DataPoint(x, y));
        }

        return points;
    }

    private static double NextGaussian(Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}

internal static class ExperimentRunner
{
    public static IReadOnlyList<ExperimentResult> Run(DatasetSplit split, ExperimentDefinition definition)
    {
        Directory.CreateDirectory(definition.OutputDirectory);
        var results = new List<ExperimentResult>();

        foreach (var optimizer in definition.Optimizers)
        {
            foreach (var learningRate in definition.LearningRates)
            {
                foreach (var l2 in definition.L2Penalties)
                {
                    var model = new LinearRegressionModel(definition.InitialWeight, definition.InitialBias);
                    var history = TrainingEngine.Train(
                        model,
                        split,
                        new TrainingConfig(
                            Epochs: definition.Epochs,
                            LearningRate: learningRate,
                            Optimizer: optimizer,
                            L2Penalty: l2,
                            EarlyStoppingPatience: definition.EarlyStoppingPatience));

                    var finalErrors = split.Validation.Select(p => model.Predict(p.X) - p.Y).ToArray();
                    var errorStats = ErrorStatistics.From(finalErrors);

                    var finalTrainLoss = history.TrainingLossByEpoch[^1];
                    var finalValLoss = history.ValidationLossByEpoch[^1];
                    var finalGradNorm = history.GradientNormByEpoch[^1];
                    var epochsRan = history.TrainingLossByEpoch.Count;
                    var gap = finalValLoss - finalTrainLoss;

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
                    MetricsLogger.Write(definition.OutputDirectory, summary, history, split.Normalization);
                }
            }
        }

        MetricsLogger.WriteLeaderboard(definition.OutputDirectory, results);
        return results;
    }
}

internal static class TrainingEngine
{
    public static TrainingHistory Train(LinearRegressionModel model, DatasetSplit split, TrainingConfig config)
    {
        var trainingLoss = new List<double>(config.Epochs);
        var validationLoss = new List<double>(config.Epochs);
        var weights = new List<double>(config.Epochs);
        var biases = new List<double>(config.Epochs);
        var gradNorms = new List<double>(config.Epochs);

        var bestValidation = double.MaxValue;
        var bestEpoch = 0;
        var bestWeight = model.Weight;
        var bestBias = model.Bias;
        var stagnantEpochs = 0;

        // Adam state.
        var mW = 0.0;
        var mB = 0.0;
        var vW = 0.0;
        var vB = 0.0;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double epsilon = 1e-8;

        for (var epoch = 1; epoch <= config.Epochs; epoch++)
        {
            var gradW = 0.0;
            var gradB = 0.0;

            foreach (var point in split.Train)
            {
                var prediction = model.Predict(point.X);
                var error = prediction - point.Y;
                gradW += error * point.X;
                gradB += error;
            }

            var trainSize = split.Train.Length;
            gradW = (2.0 / trainSize) * gradW;
            gradB = (2.0 / trainSize) * gradB;

            // L2 regularization on weight only.
            gradW += config.L2Penalty * model.Weight;

            if (config.Optimizer == OptimizerKind.Adam)
            {
                mW = beta1 * mW + (1.0 - beta1) * gradW;
                mB = beta1 * mB + (1.0 - beta1) * gradB;
                vW = beta2 * vW + (1.0 - beta2) * gradW * gradW;
                vB = beta2 * vB + (1.0 - beta2) * gradB * gradB;

                var mWHat = mW / (1.0 - Math.Pow(beta1, epoch));
                var mBHat = mB / (1.0 - Math.Pow(beta1, epoch));
                var vWHat = vW / (1.0 - Math.Pow(beta2, epoch));
                var vBHat = vB / (1.0 - Math.Pow(beta2, epoch));

                model.Weight -= config.LearningRate * mWHat / (Math.Sqrt(vWHat) + epsilon);
                model.Bias -= config.LearningRate * mBHat / (Math.Sqrt(vBHat) + epsilon);
            }
            else
            {
                model.Weight -= config.LearningRate * gradW;
                model.Bias -= config.LearningRate * gradB;
            }

            var trainLoss = CalculateMse(model, split.Train);
            var valLoss = CalculateMse(model, split.Validation);

            trainingLoss.Add(trainLoss);
            validationLoss.Add(valLoss);
            weights.Add(model.Weight);
            biases.Add(model.Bias);
            gradNorms.Add(Math.Sqrt(gradW * gradW + gradB * gradB));

            if (valLoss < bestValidation)
            {
                bestValidation = valLoss;
                bestEpoch = epoch;
                bestWeight = model.Weight;
                bestBias = model.Bias;
                stagnantEpochs = 0;
            }
            else
            {
                stagnantEpochs++;
            }

            if (stagnantEpochs >= config.EarlyStoppingPatience)
            {
                break;
            }
        }

        // Restore best checkpoint after early stopping.
        model.Weight = bestWeight;
        model.Bias = bestBias;

        return new TrainingHistory(trainingLoss, validationLoss, weights, biases, gradNorms, bestEpoch);
    }

    private static double CalculateMse(LinearRegressionModel model, IReadOnlyList<DataPoint> points)
    {
        var squaredError = 0.0;

        foreach (var point in points)
        {
            var error = model.Predict(point.X) - point.Y;
            squaredError += error * error;
        }

        return squaredError / points.Count;
    }
}

internal static class MetricsLogger
{
    public static void Write(string outputDirectory, ExperimentResult summary, TrainingHistory history, Normalization normalization)
    {
        var runId = BuildRunId(summary.Optimizer, summary.LearningRate, summary.L2Penalty);
        var csvPath = Path.Combine(outputDirectory, $"run_{runId}.csv");
        var jsonPath = Path.Combine(outputDirectory, $"run_{runId}.json");

        var csv = new StringBuilder();
        csv.AppendLine("epoch,training_loss,validation_loss,weight,bias,gradient_norm");

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

    public static void WriteLeaderboard(string outputDirectory, IEnumerable<ExperimentResult> results)
    {
        var ordered = results.OrderBy(r => r.FinalValidationLoss).ToArray();
        var path = Path.Combine(outputDirectory, "leaderboard.csv");

        var csv = new StringBuilder();
        csv.AppendLine("rank,optimizer,learning_rate,l2_penalty,final_training_loss,final_validation_loss,generalization_gap,final_gradient_norm,best_epoch,epochs_ran,convergence_status,final_weight,final_bias,error_mean,error_std_dev,error_p50,error_p90");

        for (var i = 0; i < ordered.Length; i++)
        {
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

    private static string BuildRunId(OptimizerKind optimizer, double learningRate, double l2)
    {
        var lrSlug = learningRate.ToString("0.###", CultureInfo.InvariantCulture).Replace('.', '_');
        var l2Slug = l2.ToString("0.###", CultureInfo.InvariantCulture).Replace('.', '_');
        return $"opt_{optimizer}_lr_{lrSlug}_l2_{l2Slug}".ToLowerInvariant();
    }
}

internal static class RunDiagnostics
{
    public static string AssessConvergence(double gradientNorm, double generalizationGap, int bestEpoch, int epochsRan)
    {
        if (gradientNorm > 0.1)
        {
            return "underfit";
        }

        if (generalizationGap > 0.15)
        {
            return "overfit";
        }

        if (bestEpoch < epochsRan - 15)
        {
            return "plateau";
        }

        return "stable";
    }
}

internal static class ErrorStatistics
{
    public static ErrorSummary From(IReadOnlyList<double> errors)
    {
        var ordered = errors.OrderBy(e => e).ToArray();
        var mean = ordered.Average();
        var variance = ordered.Average(e => (e - mean) * (e - mean));

        return new ErrorSummary(
            Mean: mean,
            StdDev: Math.Sqrt(variance),
            P50: Quantile(ordered, 0.50),
            P90: Quantile(ordered, 0.90));
    }

    private static double Quantile(IReadOnlyList<double> ordered, double q)
    {
        if (ordered.Count == 0)
        {
            return 0.0;
        }

        var index = (ordered.Count - 1) * q;
        var lower = (int)Math.Floor(index);
        var upper = (int)Math.Ceiling(index);

        if (lower == upper)
        {
            return ordered[lower];
        }

        var weight = index - lower;
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight;
    }
}

internal sealed class LinearRegressionModel(double initialWeight, double initialBias)
{
    public double Weight { get; set; } = initialWeight;

    public double Bias { get; set; } = initialBias;

    public double Predict(double x) => Weight * x + Bias;
}

internal enum OptimizerKind
{
    Sgd,
    Adam
}

internal readonly record struct DataPoint(double X, double Y);

internal readonly record struct Normalization(double Mean, double StdDev)
{
    public static Normalization None => new(0.0, 1.0);
}

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

internal sealed record DatasetSplit(DataPoint[] Train, DataPoint[] Validation, Normalization Normalization);

internal sealed record ExperimentDefinition(
    int Epochs,
    int EarlyStoppingPatience,
    double InitialWeight,
    double InitialBias,
    double[] LearningRates,
    OptimizerKind[] Optimizers,
    double[] L2Penalties,
    string OutputDirectory);

internal sealed record TrainingConfig(
    int Epochs,
    double LearningRate,
    OptimizerKind Optimizer,
    double L2Penalty,
    int EarlyStoppingPatience);

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

internal sealed record ErrorSummary(double Mean, double StdDev, double P50, double P90);

internal sealed record TrainingHistory(
    IReadOnlyList<double> TrainingLossByEpoch,
    IReadOnlyList<double> ValidationLossByEpoch,
    IReadOnlyList<double> WeightByEpoch,
    IReadOnlyList<double> BiasByEpoch,
    IReadOnlyList<double> GradientNormByEpoch,
    int BestEpoch);
