using Microsoft.ML.Data;

public class ProductRating
    {
        [LoadColumn(0)] public float UserId;
        [LoadColumn(1)] public float ProductId;
        [LoadColumn(2)] public bool Label;
    }
