using Microsoft.ML;

public class DataLoader
{
    private readonly MLContext _mlContext;

    public DataLoader(MLContext mlContext)
    {
        _mlContext = mlContext;
    }

    public IDataView LoadData(string purchasesPath, string purchaseDetailsPath)
    {
        var purchases = File.ReadAllLines(purchasesPath).Skip(1)
            .Select(line => line.Split(','))
            .ToDictionary(parts => int.Parse(parts[0]), parts => int.Parse(parts[1]));

        var details = File.ReadAllLines(purchaseDetailsPath).Skip(1)
            .Select(line => line.Split(','))
            .Select(parts => new
            {
                PurchaseId = int.Parse(parts[0]),
                ProductName = parts[1],
                Quantity = float.Parse(parts[2])
            });

        var productToId = details.Select(d => d.ProductName).Distinct()
            .Select((name, index) => new { Name = name, Id = index + 1 })
            .ToDictionary(x => x.Name, x => x.Id);

        var trainingData = details
            .Join(purchases, d => d.PurchaseId, c => c.Key, (d, c) => new ProductRating
            {
                UserId = c.Value,
                ProductId = productToId[d.ProductName],
                Label = d.Quantity > 0
            })
            .ToList();

        return _mlContext.Data.LoadFromEnumerable(trainingData);
    }

    public Dictionary<int, string> LoadProducts(string purchaseDetailsPath)
    {
        var products = File.ReadAllLines(purchaseDetailsPath).Skip(1)
            .Select(line => line.Split(',')[1])
            .Distinct()
            .Select((name, index) => new { Name = name, Id = index + 1 })
            .ToDictionary(x => x.Id, x => x.Name);

        return products;
    }
}
