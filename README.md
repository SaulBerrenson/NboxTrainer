# NBox Trainer
Nbox Trainer is library-wrapper for easy creating trainer ml models of image classification with GPU acceleration. Also you can test prediction with prediction service.

#### Packages
* Microsoft.ML [1.5.4]
* Microsoft.ML.ImageAnalytics [1.5.4]
* Microsoft.ML.Vision [1.5.4]
* SciSharp.TensorFlow.Redist-Windows-GPU [1.5.1]
* CUDA 10.0;
* Cudnn 7.6.4;

###### Installing cuda and cudnn
https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/install-gpu-model-builder


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

#### Example using trainer with fluent builder
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


#### Example using prediction service with fluent builder
```c#
   PredictorService service = new PredictorService("D:\\flower_photos\\test_fraction\\")
                        .setPathToModel("D:\\21-01-10_flower.zip");
            service.onPrediction += prediction => Console.WriteLine(prediction);
            await service.Process();
            
/* output log with event onPrediction
*
*File:   100080576_f52e8ee070_n.jpg |  Predicted: daisy  |    Accuracy: 0,9956813
*File:   102841525_bd6628ae3c.jpg   |  Predicted: daisy  |    Accuracy: 0,99993706
*File:   105806915_a9c13e2106_n.jpg |  Predicted: daisy  |    Accuracy: 0,99083275
*File:   107592979_aaa9cdfe78_m.jpg |  Predicted: daisy  |    Accuracy: 0,9820342
*File:   113902743_8f537f769b_n.jpg |  Predicted: tulips |    Accuracy: 0,9946406
*File:   113960470_38fab8f2fb_m.jpg |  Predicted: tulips |    Accuracy: 0,9788831
*File:   116343334_9cb4acdc57_n.jpg |  Predicted: tulips |    Accuracy: 0,99646
*File:   122450705_9885fff3c4_n.jpg |  Predicted: tulips |    Accuracy: 0,9952939
*/
```



