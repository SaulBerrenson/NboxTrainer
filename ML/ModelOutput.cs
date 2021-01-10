using System.IO;
using System.Linq;

namespace NboxTrainer.ML
{
    public class ModelOutput
    {
        public string ImagePath { get; set; }

        public string Label { get; set; }

        public string PredictedLabel { get; set; }

        public float[] Score { get; set; }

        public override string ToString()
        {
            return $"File:   {new FileInfo(ImagePath)?.Name} |  Predicted: {PredictedLabel}  |    Accuracy: {Score?.Max()}";
        }
    }
}