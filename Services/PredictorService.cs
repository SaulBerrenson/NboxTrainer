using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using NboxTrainer.Helper;
using NboxTrainer.ML;

namespace NboxTrainer.Services
{
    public class PredictorService
    {
        private string targetsDir;
        private string _modelPath;
        private MLContext _mlContext;
        private ServiceStatePredictor _statePredictor;

        public delegate void OnPrediction(ModelOutput prediction);
        public event OnPrediction onPrediction;

        public PredictorService(string targetsDir)
        {
            this.targetsDir = targetsDir;
            _mlContext = new MLContext();
            _statePredictor = new ServiceStatePredictor();
        }

        public PredictorService(string targetsDir, string modelPath)
        {
            if(!FileIO.dirExist(targetsDir)) throw new Exception("TARGER DIR NOT EXIST");
            if(!FileIO.fileExist(modelPath)) throw new Exception("MODEL FILE NOT EXIST");

            this.targetsDir = targetsDir;
            _modelPath = modelPath;
            _mlContext = new MLContext();
            _statePredictor = new ServiceStatePredictor();
        }

        public PredictorService setPathToModel(string path)
        {
            if (!FileIO.fileExist(path)) throw new Exception("MODEL FILE NOT EXIST");
            _modelPath = path;
            return this;
        }

        public async Task Process()
        {
            _statePredictor.setState(StatePrediction.Running);
            await AsyncIO.StartTask(() =>
            {
                try
                {
                    var model_prediction = _mlContext.Model.Load(_modelPath, out var schema);
                    var predictor = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model_prediction, schema);

                    foreach (var imageData in FileIO.loadImagesFromDirectory(targetsDir))
                    {
                        if(imageData==null) continue;
                        imageData.Image = File.ReadAllBytes(imageData.ImagePath);
                        onPrediction?.Invoke(predictor.Predict(imageData));
                    }

                }
                catch (Exception e)
                {
                    
                    throw e;
                }

            });
            _statePredictor.setState(StatePrediction.Done);
        }
    }
}