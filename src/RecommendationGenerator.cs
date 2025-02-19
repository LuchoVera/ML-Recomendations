using Microsoft.ML;

public class RecommendationGenerator
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<ProductRating, ProductRatingPrediction> _predictionEngine;

    public RecommendationGenerator(MLContext mlContext, ITransformer trainedModel)
    {
        _mlContext = mlContext;
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ProductRating, ProductRatingPrediction>(trainedModel);
    }

    public List<Tuple<string, float>> GenerateRecommendationsForUser(int userId, Dictionary<int, string> productsById)
    {
        var recommendations = new List<Tuple<string, float>>();

        foreach (var product in productsById)
        {
            var prediction = _predictionEngine.Predict(new ProductRating
            {
                UserId = userId,
                ProductId = product.Key
            });

            recommendations.Add(new Tuple<string, float>(product.Value, prediction.Score));
        }

        return recommendations.OrderByDescending(x => x.Item2).ToList();
    }
}
