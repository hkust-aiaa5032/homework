# Instructions for hw2

In this homework we will perform a video classification task using CNN.

## Data and Labels

Please download data from kaggle withi this [link](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw2/data).

## Step-by-step baseline instructions

We give you a video folder, video_frames_30fpv_320p. Under each sub folder, there are 30 images from one video, we use the middle image for classification.

### CNN classifier

We use CNN as classifier.

Suppose you are under `hw2` directory. Train CNN by:

```
$ python CNN_Classifier_train.py
```

Run Model Inference on the test set:

```
$ python CNN_Classifier_test2csv.py
```


### Submission to Kaggle

You can then submit the test outputs to the leaderboard:

```
https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw2/
```

We use accuracy as the evaluation metric. Please refer to `**test_for_student.csv**` for submission format.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Try different data augmentation method
+ Try different split ratio of train and valition
+ Try diffirent Neural Network Design
+ Try diffirent overfitting methods(BN, dropout, Early Stopping)
+ Try different fusion or model aggregation methods. For example, you can simply average two model predictions (late fusion)

