using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using NboxTrainer.Helper;
using NboxTrainer.ML;
using static Microsoft.ML.Vision.ImageClassificationTrainer;

namespace NboxTrainer.Services
{
    /// <summary>
    /// Class for creating pipeline image classification
    /// </summary>
    public class TrainerService : ITrainer
    {

        #region private
        private string dir_dataset;
        private Dictionary<string,int> list_categories = new Dictionary<string, int>();
        private IDataView _preProcessedData;
        private float _fractionDataset;
        private MLContext _mlContext;
        private IDataView _trainSet;
        private IDataView _validationSet;
        private IDataView _testSet;
        private Options _trainOptions;
        private string project_directory;
        private string project_name;
        private ITransformer _trainedModel;
        private readonly ServiceStateML _serviceStateMl = new ServiceStateML();
        #endregion
        #region events

        public delegate void OnServiceState(StateML state);
        public event OnServiceState onStateChanged;

        public delegate void OnTrainMetrics(ImageClassificationMetrics.Dataset dataset,
            ImageClassificationMetrics metrics);
        public event OnTrainMetrics onTrainMetrics;

        #endregion

        #region contructor

        public TrainerService(string projectDirectory, string projectName)
        {
            project_directory = projectDirectory;
            project_name = projectName;

            _mlContext = new MLContext();

            _serviceStateMl.onStateChanged += state => onStateChanged?.Invoke(state); 

            _trainOptions = new Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                Epoch = 500,
                LearningRate = 0.01F,
                Arch = Architecture.MobilenetV2,
                BatchSize = 100,
                ReuseValidationSetBottleneckCachedValues = true,
                ReuseTrainSetBottleneckCachedValues = true,
                TrainSetBottleneckCachedValuesFileName = Path.Combine(projectDirectory, "train_bottleneck_cache.csv"),
                ValidationSetBottleneckCachedValuesFileName = Path.Combine(projectDirectory, "validation_bottleneck_cache.csv"),
                ValidationSet = _validationSet,
                WorkspacePath = projectDirectory,
                MetricsCallback = (metrics) =>
                {
                    if (metrics.Bottleneck != null) { _serviceStateMl.setState(StateML.Bottleneck); }
                    if (metrics.Train != null) { _serviceStateMl.setState(StateML.Train);  onTrainMetrics?.Invoke(metrics.Train.DatasetUsed,metrics);}
                },
            };

           
        }

        #endregion



        #region prepare dataset

        private void splitDatasets()
        {
            if(_fractionDataset<=0) throw new Exception("FRACTION IS ZERO OR BELOW");

            DataOperationsCatalog.TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(data: _preProcessedData, testFraction: _fractionDataset);
            DataOperationsCatalog.TrainTestData validationTestSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);
            _trainSet = trainSplit.TrainSet;
            _validationSet = validationTestSplit.TrainSet;
            _testSet = validationTestSplit.TestSet;
            _trainOptions.ValidationSet = _validationSet;
            _trainOptions.TrainSetBottleneckCachedValuesFileName =
                Path.Combine(AppContext.BaseDirectory, "train_bottleneck_cache.csv");
            _trainOptions.ValidationSetBottleneckCachedValuesFileName =
                Path.Combine(AppContext.BaseDirectory, "validation_bottleneck_cache.csv");
        }

        private async Task<bool> loadDataset()
        {
            if (!FileIO.dirExist(dir_dataset)) return false;


            try
            {
                foreach (var cat_dir in FileIO.getListDirectiories(dir_dataset))
                {
                    await AsyncIO.StartTask(() =>
                    {
                        var name_category = new DirectoryInfo(cat_dir).Name;
                        int count = FileIO.countFiles(cat_dir);

                        if (list_categories.ContainsKey(name_category)) list_categories[name_category] = count;
                        else
                        {
                            list_categories.Add(name_category, count);
                        }
                    });
                }

                IEnumerable<ImageData> images = FileIO.loadImagesFromDirectory(folder: dir_dataset, useFolderNameAsLabel: true);
                // IDataView fullImagesDataset = _mlContext.Data.LoadFromEnumerable(images);
                // IDataView shuffledFullImageFilePathsDataset = _mlContext.Data.ShuffleRows(fullImagesDataset);

                IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);

                IDataView shuffledData = _mlContext.Data.ShuffleRows(imageData);

                var preprocessingPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                        inputColumnName: "Label",
                        outputColumnName: "LabelAsKey")
                    .Append(_mlContext.Transforms.LoadRawImageBytes(
                        outputColumnName: "Image",
                        imageFolder: dir_dataset,
                        inputColumnName: "ImagePath"));


                _preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);
            }
            catch (Exception e)
            {
                return false;
            }
            return true;
        }

        #endregion

        ///////////////////////////////
        /// TrainOptions
        //////////////////////////////

        #region fluent builder settings

        public TrainerService setPathToDataset(string dir)
        {
            list_categories?.Clear();
            foreach (var catName in FileIO.getListNamesDirectiories(dir))
            {
                list_categories?.Add(catName, -1);
            }
            dir_dataset = dir;
            return this;
        }
        public TrainerService setSplitPercent(float fractionDataset = 0.1F)
        {
            _fractionDataset = fractionDataset;
            return this;
        }

        public TrainerService setOptions(Options options)
        {
            _trainOptions = options;
            return this;
        }
        public TrainerService setCriteriaEarlyStopping(EarlyStopping criteria = null)
        {
            if (criteria == null) criteria = new EarlyStopping(0.01F, 20, EarlyStoppingMetric.Accuracy, true);
            _trainOptions.EarlyStoppingCriteria = criteria;
            return this;
        }

        public TrainerService setReuseTrainBottleneckCache(bool reuse = true)
        {
            _trainOptions.ReuseTrainSetBottleneckCachedValues = reuse;
            return this; 
        }

        public TrainerService setReuseValidationBottleneckCache(bool reuse = true)
        {
            _trainOptions.ReuseValidationSetBottleneckCachedValues = reuse;
            return this;
        }

        public TrainerService setEpoch(int epoch = 40, int patience = 10)
        {
            _trainOptions.Epoch = epoch;
            _trainOptions.EarlyStoppingCriteria.Patience = patience;
            return this;
        }

        public TrainerService setArchitecture(Architecture architecture = Architecture.MobilenetV2)
        {
            _trainOptions.Arch = architecture;
            return this;
        }

        public TrainerService setBatchSize(int batchSize = 10)
        {
            _trainOptions.BatchSize = batchSize;
            return this;
        }

        public TrainerService setLearningRate(float learningRate = 0.01f)
        {
            _trainOptions.LearningRate = learningRate;
            return this;
        }

        public TrainerService setProjectName(string name, string working_directory)
        {
            project_directory = working_directory;
            project_name = name;
            _trainOptions.WorkspacePath = working_directory;
            _trainOptions.FinalModelPrefix = name;
            return this;
        }

        #endregion

        ///////////////////////////////
        /// Global Methods
        //////////////////////////////


        #region main methods

        public async Task<bool> Train()
        { 
            bool result = false;

            await loadDataset();
            splitDatasets();

            await AsyncIO.StartTask(() =>
            {
                try
                {
                    var pipeline = _mlContext.MulticlassClassification.Trainers.ImageClassification(_trainOptions)
                        .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                            outputColumnName: "PredictedLabel",
                            inputColumnName: "PredictedLabel"));
                    _trainedModel = pipeline.Fit(_trainSet);
                    result = true;
                  
                }
                catch (Exception e)
                {
                    result = false;
                }
               
            });

            _serviceStateMl.setState(StateML.Done);
            return result;
        }

        public async Task<MulticlassClassificationMetrics> GetMetrics()
        {
            MulticlassClassificationMetrics metrics = null;

            await AsyncIO.StartTask(() =>
            {
                IDataView predictionsDataView = _trainedModel.Transform(_testSet);
                metrics = _mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            });

            return metrics;
        }


        public async Task SaveModel()
        {
            await AsyncIO.StartTask(() => _mlContext.Model.Save(_trainedModel, _trainSet.Schema,
                Path.Combine(AppContext.BaseDirectory, $"{DateTime.Now:yy-MM-dd}_{project_name}.zip")));
        }

        #endregion

    }
}