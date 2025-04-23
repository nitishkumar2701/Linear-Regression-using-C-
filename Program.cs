using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LinearRegressionExample
{
    class Program
    {
        // Define paths
        private static string DataPath = "../Linear-Regression-using-Dotnet/test_energy_data.csv";
        
        // Define the data structure (modify these fields to match your Kaggle dataset columns)
        public class InputData
        {
            
            [LoadColumn(1)] //Column Index in file
            public float SquareFootage { get; set; }
            
            [LoadColumn(2)]
            public float NumberOfOccupants { get; set; }
            
            [LoadColumn(3)]
            public float ApplicancesUsed { get; set; } 

            [LoadColumn(4)]
            public float AverageTemperature { get; set; }
            
            [LoadColumn(6)]
            public float EnergyConsumption { get; set; }
        }

        // Define the prediction class
        public class Prediction
        {
            [ColumnName("Score")]
            public float PredictedValue { get; set; }
        }

        static void Main(string[] args)
        {
            // Create ML context
            var mlContext = new MLContext(seed: 0);

            // Check if file exists
            if (!File.Exists(DataPath))
            {
                Console.WriteLine($"Error: File not found at {Path.GetFullPath(DataPath)}");
                return;
            }

            // Load data from CSV file
            Console.WriteLine("Loading data...");
            IDataView dataView = mlContext.Data.LoadFromTextFile<InputData>(
                path: DataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true);
                
            // Split data
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            
            // Define features for training - use property names, not column headers
            var featureColumns = new[] { "SquareFootage", "NumberOfOccupants", "ApplicancesUsed", "AverageTemperature" };
            
            // Create pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "EnergyConsumption"));
                
            // Train the model
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(trainTestSplit.TrainSet);
            
            // Evaluate
            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "EnergyConsumption");
            
            // Display metrics
            Console.WriteLine($"R-Squared: {metrics.RSquared:0.###}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:0.###}");
            
            // Make a prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, Prediction>(model);
            
            // Create a sample input with actual values
            var sampleData = new InputData { 
                SquareFootage = 2000, 
                NumberOfOccupants = 3, 
                ApplicancesUsed = 10, 
                AverageTemperature = 72 
            };
            
            var prediction = predictionEngine.Predict(sampleData);
            
            Console.WriteLine($"Predicted energy consumption for a 2000 sq ft home with 3 occupants, 10 appliances, and 72°F average temperature: {prediction.PredictedValue:0.###} units");
            
            // Save model
            // mlContext.Model.Save(model, dataView.Schema, "EnergyModel.zip");
            // Console.WriteLine("Model saved as EnergyModel.zip");
        }
    }
}