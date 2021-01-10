# NBox Trainer
Nbox Trainer is library-wrapper for easy creating trainer ml models of image classification with GPU acceleration. Used ML.Net and Tensorflow

#### Packages
* Microsoft.ML [1.5.4]
* Microsoft.ML.ImageAnalytics [1.5.4]
* Microsoft.ML.Vision [1.5.4]
* SciSharp.TensorFlow.Redist-Windows-GPU [1.5.1]
* CUDA 10.0;
* Cudnn 7.6.4;

#### Example struct dataset
```
 +---flower_photos
 |   +---daisy
 |   +---dandelion
 |   +---roses
 |   +---sunflowers
 |   +---tulips
 |--------------------
```

All top subfolders of dataset catalog will use as label of categories

#### Example using with fluent builder
```c#
 string dirDatasets = "D:\\Downloads\\flower_photos\\";

            TrainerService trainer = new TrainerService(AppContext.BaseDirectory, "flower")
                .setArchitecture(ImageClassificationTrainer.Architecture.MobilenetV2)
                .setEpoch(1000)
                .setBatchSize(5)
                .setLearningRate(0.01F)
                .setReuseTrainBottleneckCache(true)
                .setReuseValidationBottleneckCache(true)
                .setCriteriaEarlyStopping(new ImageClassificationTrainer.EarlyStopping(0.1F, 500,
                    ImageClassificationTrainer.EarlyStoppingMetric.Accuracy))
                .setPathToDataset(dirDatasets)
                .setSplitPercent(0.3F);

            trainer.onStateChanged += state => { Console.Title = $"Current Stage: {state}"; };
            trainer.onTrainMetrics += Trainer_onTrainMetrics;

            await trainer.Train();
            var _metrics = await trainer.GetMetrics();
            await trainer.SaveModel();
```
