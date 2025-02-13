using Microsoft.ML;

namespace RecommendationSystem
{
    class Program
    {
        private static string PurchasesPath = "resources/compras.csv";
        private static string PurchaseDetailsPath = "resources/detalles_compras.csv";

        static void Main(string[] args)
        {
            try
            {
                var mlContext = new MLContext();
                var dataLoader = new DataLoader(mlContext);
                var modelTrainer = new ModelTrainer(mlContext);

                Console.WriteLine("Loading data");
                IDataView trainData = dataLoader.LoadData(PurchasesPath, PurchaseDetailsPath);

                Console.WriteLine("Training the model");
                ITransformer trainedModel = modelTrainer.TrainModel(trainData);

                Console.WriteLine("Generating recommendations for user");
                var recommendationGenerator = new RecommendationGenerator(mlContext, trainedModel);
                var productsById = dataLoader.LoadProducts(PurchaseDetailsPath);

                var recommendations = recommendationGenerator.GenerateRecommendationsForUser(45, productsById);

                Console.WriteLine("Top 5 recommended products:");
                foreach (var recommendation in recommendations.Take(5))
                {
                    Console.WriteLine($"- {recommendation.Item1}: Score {recommendation.Item2:F2}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }

}
