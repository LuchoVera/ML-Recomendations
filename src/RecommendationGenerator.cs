using Microsoft.ML;

public class RecommendationGenerator
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<ProductRating, ProductRatingPrediction> _predictionEngine;
    private readonly Dictionary<int, HashSet<int>> _userPurchaseHistory;

    public RecommendationGenerator(
        MLContext mlContext, 
        ITransformer trainedModel, 
        string purchasesPath, 
        string purchaseDetailsPath)
    {
        _mlContext = mlContext;
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ProductRating, ProductRatingPrediction>(trainedModel);
        _userPurchaseHistory = BuildUserPurchaseHistory(purchasesPath, purchaseDetailsPath);
    }

    //Construye un historial de compras para cada usuario.
    private Dictionary<int, HashSet<int>> BuildUserPurchaseHistory(string purchasesPath, string purchaseDetailsPath)
    {
        var result = new Dictionary<int, HashSet<int>>();
        
        var purchases = File.ReadAllLines(purchasesPath).Skip(1)
            .Select(line => line.Split(','))
            .ToDictionary(parts => int.Parse(parts[0]), parts => int.Parse(parts[1]));
        
        var productDetails = File.ReadAllLines(purchaseDetailsPath).Skip(1)
            .Select(line => line.Split(','))
            .Select(parts => new {
                PurchaseId = int.Parse(parts[0]),
                ProductName = parts[1],
                Quantity = float.Parse(parts[2])
            });
            
        var productToId = productDetails.Select(d => d.ProductName).Distinct()
            .Select((name, index) => new { Name = name, Id = index + 1 })
            .ToDictionary(x => x.Name, x => x.Id);
            
        foreach (var detail in productDetails)
        {
            if (purchases.TryGetValue(detail.PurchaseId, out int userId))
            {
                int productId = productToId[detail.ProductName];
                
                if (!result.ContainsKey(userId))
                {
                    result[userId] = new HashSet<int>();
                }
                
                result[userId].Add(productId);
            }
        }
        
        return result;
    }

    //Genera recomendaciones para un usuario dado
    public List<Tuple<string, float>> GenerateRecommendationsForUser(int userId, Dictionary<int, string> productsById)
    {
        var recommendations = new List<Tuple<string, float>>();
        HashSet<int> userPurchasedProducts = new HashSet<int>();
        
        if (_userPurchaseHistory.TryGetValue(userId, out var purchasedProducts))
        {
            userPurchasedProducts = purchasedProducts;
        }

        foreach (var product in productsById)
        {
            if (userPurchasedProducts.Contains(product.Key))
            {
                continue;
            }
            
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
