using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MyMLLab;

internal static class Program
{
    private static void Main()
    {
        var dataset = DatasetEngine.GenerateLinear(
            count: 180,
            slope: 2.5,
            intercept: -0.8,
            minX: -2.0,
            maxX: 2.0,
            noiseStdDev: 0.2,
            seed: 42);

        var split = DatasetEngine.Split(dataset, trainRatio: 0.8);

        var experiment = new ExperimentDefinition(
            Epochs: 120,
            LearningRates: new[] { 0.001, 0.01, 0.05, 0.1 },
            InitialWeight: 0.0,
            InitialBias: 0.0,
            OutputDirectory: "artifacts");

        var results = ExperimentRunner.Run(split, experiment);

        Console.WriteLine("MyMLLab experiment results (ranked by final validation loss):");
        foreach (var result in results.OrderBy(r => r.FinalValidationLoss))
        {
            Console.WriteLine($"lr={result.LearningRate,-6:0.###} train={result.FinalTrainingLoss,10:0.000000} val={result.FinalValidationLoss,10:0.000000} w={result.FinalWeight,8:0.0000} b={result.FinalBias,8:0.0000}");
        }

        Console.WriteLine($"\nArtifacts written to: {Path.GetFullPath(experiment.OutputDirectory)}");
    }
}

internal static class DatasetEngine
{
    public static IReadOnlyList<DataPoint> GenerateLinear(int count, double slope, double intercept, double minX, double maxX, double noiseStdDev, int seed)
    {
        var random = new Random(seed);
        var points = new List<DataPoint>(capacity: count);

        for (var i = 0; i < count; i++)
        {
            var x = minX + (maxX - minX) * random.NextDouble();
            var noise = NextGaussian(random) * noiseStdDev;
            var y = slope * x + intercept + noise;
            points.Add(new DataPoint(x, y));
        }

        return points;
    }

    public static DatasetSplit Split(IReadOnlyList<DataPoint> dataset, double trainRatio)
    {
        var trainCount = (int)(dataset.Count * trainRatio);
        var train = dataset.Take(trainCount).ToArray();
        var validation = dataset.Skip(trainCount).ToArray();
        return new DatasetSplit(train, validation);
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
        var results = new List<ExperimentResult>(definition.LearningRates.Length);

        foreach (var learningRate in definition.LearningRates)
        {
            var model = new LinearRegressionModel(definition.InitialWeight, definition.InitialBias);
            var history = TrainingEngine.Train(model, split, learningRate, definition.Epochs);
            var summary = new ExperimentResult(
                learningRate,
                history.TrainingLossByEpoch[^1],
                history.ValidationLossByEpoch[^1],
                history.WeightByEpoch[^1],
                history.BiasByEpoch[^1]);

            results.Add(summary);
            MetricsLogger.Write(definition.OutputDirectory, summary, history);
        }

        MetricsLogger.WriteLeaderboard(definition.OutputDirectory, results);
        return results;
    }
}

internal static class TrainingEngine
{
    public static TrainingHistory Train(LinearRegressionModel model, DatasetSplit split, double learningRate, int epochs)
    {
        var trainingLoss = new List<double>(epochs);
        var validationLoss = new List<double>(epochs);
        var weights = new List<double>(epochs);
        var biases = new List<double>(epochs);

        for (var epoch = 1; epoch <= epochs; epoch++)
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

            model.Weight -= learningRate * gradW;
            model.Bias -= learningRate * gradB;

            trainingLoss.Add(CalculateMse(model, split.Train));
            validationLoss.Add(CalculateMse(model, split.Validation));
            weights.Add(model.Weight);
            biases.Add(model.Bias);
        }

        return new TrainingHistory(trainingLoss, validationLoss, weights, biases);
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
    public static void Write(string outputDirectory, ExperimentResult summary, TrainingHistory history)
    {
        var lrSlug = summary.LearningRate.ToString("0.###", CultureInfo.InvariantCulture).Replace('.', '_');
        var csvPath = Path.Combine(outputDirectory, $"run_lr_{lrSlug}.csv");
        var jsonPath = Path.Combine(outputDirectory, $"run_lr_{lrSlug}.json");

        var csv = new StringBuilder();
        csv.AppendLine("epoch,training_loss,validation_loss,weight,bias");

        for (var i = 0; i < history.TrainingLossByEpoch.Count; i++)
        {
            csv.AppendLine(string.Join(',',
                i + 1,
                history.TrainingLossByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.ValidationLossByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.WeightByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture),
                history.BiasByEpoch[i].ToString("0.000000", CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(csvPath, csv.ToString());

        var payload = new
        {
            summary.LearningRate,
            summary.FinalTrainingLoss,
            summary.FinalValidationLoss,
            summary.FinalWeight,
            summary.FinalBias,
            history.TrainingLossByEpoch,
            history.ValidationLossByEpoch,
            history.WeightByEpoch,
            history.BiasByEpoch
        };

        File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static void WriteLeaderboard(string outputDirectory, IEnumerable<ExperimentResult> results)
    {
        var ordered = results.OrderBy(r => r.FinalValidationLoss).ToArray();
        var path = Path.Combine(outputDirectory, "leaderboard.csv");

        var csv = new StringBuilder();
        csv.AppendLine("rank,learning_rate,final_training_loss,final_validation_loss,final_weight,final_bias");

        for (var i = 0; i < ordered.Length; i++)
        {
            var row = ordered[i];
            csv.AppendLine(string.Join(',',
                i + 1,
                row.LearningRate.ToString("0.###", CultureInfo.InvariantCulture),
                row.FinalTrainingLoss.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalValidationLoss.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalWeight.ToString("0.000000", CultureInfo.InvariantCulture),
                row.FinalBias.ToString("0.000000", CultureInfo.InvariantCulture)));
        }

        File.WriteAllText(path, csv.ToString());
    }
}

internal sealed class LinearRegressionModel(double initialWeight, double initialBias)
{
    public double Weight { get; set; } = initialWeight;

    public double Bias { get; set; } = initialBias;

    public double Predict(double x) => Weight * x + Bias;
}

internal readonly record struct DataPoint(double X, double Y);

internal sealed record DatasetSplit(DataPoint[] Train, DataPoint[] Validation);

internal sealed record ExperimentDefinition(int Epochs, double[] LearningRates, double InitialWeight, double InitialBias, string OutputDirectory);

internal sealed record ExperimentResult(double LearningRate, double FinalTrainingLoss, double FinalValidationLoss, double FinalWeight, double FinalBias);

internal sealed record TrainingHistory(IReadOnlyList<double> TrainingLossByEpoch, IReadOnlyList<double> ValidationLossByEpoch, IReadOnlyList<double> WeightByEpoch, IReadOnlyList<double> BiasByEpoch);
