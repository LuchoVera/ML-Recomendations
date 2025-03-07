using Microsoft.ML;

public class ModelTrainer
{
    private readonly MLContext _mlContext;

    public ModelTrainer(MLContext mlContext)
    {
        _mlContext = mlContext;
    }

    public ITransformer TrainModel(IDataView trainData)
    {
        //Define un pipeline de procesamiento y entrena el modelo
        // Codifica IDs de usuario y producto
        var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: "UserId")
            .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "ProductIdEncoded", inputColumnName: "ProductId"))
            // Aplica codificación one-hot
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "UserIdFeatures", inputColumnName: "UserIdEncoded"))
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ProductIdFeatures", inputColumnName: "ProductIdEncoded"))
            .Append(_mlContext.Transforms.Concatenate("Features", "UserIdFeatures", "ProductIdFeatures"))
            // Entrena usando FieldAwareFactorizationMachine
            .Append(_mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(
                labelColumnName: "Label",
                featureColumnName: "Features"));

        return pipeline.Fit(trainData);
    }
}
