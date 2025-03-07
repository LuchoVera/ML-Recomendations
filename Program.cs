class Program
{
    private static string PurchasesPath = "resources/compras.csv";
    private static string PurchaseDetailsPath = "resources/detalles_compras.csv";
    static void Main(string[] args)
    {
        RecommendationPrinter recommendationPrinter= new RecommendationPrinter();
        recommendationPrinter.GenerateRecomendations(3, PurchasesPath, PurchaseDetailsPath);
    }
}
