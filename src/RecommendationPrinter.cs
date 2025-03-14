using Microsoft.ML;
public class RecommendationPrinter
{
    public void GenerateRecomendations(int userId, string purchasesPath, string purchaseDetailsPath)
    {
        try
        {
            var mlContext = new MLContext();
            var dataLoader = new DataLoader(mlContext);
            var modelTrainer = new ModelTrainer(mlContext);

            Console.WriteLine("Loading data");
            IDataView trainData = dataLoader.LoadData(purchasesPath, purchaseDetailsPath);

            Console.WriteLine("Training the model");
            ITransformer trainedModel = modelTrainer.TrainModel(trainData);

            Console.WriteLine("Generating recommendations for user");
            var recommendationGenerator = new RecommendationGenerator(
                mlContext, 
                trainedModel, 
                purchasesPath, 
                purchaseDetailsPath);
                
            var productsById = dataLoader.LoadProducts(purchaseDetailsPath);

            var recommendations = recommendationGenerator.GenerateRecommendationsForUser(userId, productsById);

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
