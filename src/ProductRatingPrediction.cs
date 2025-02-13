using Microsoft.ML.Data;

public class ProductRatingPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
    }
