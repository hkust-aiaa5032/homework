# Instructions for hw3

In this homework we will perform a video classification task using 3D CNN.

## Data and Labels

Please download data from kaggle with this [link](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw3-spring-2024/data).

## Step-by-step baseline instructions

We give you a video folder, **hw3_16fpv**. Under each sub folder, there are 16 images from one video, we use these 16 image to form a 5D Tensor(N C D H W). We also provide you the origon mp4 videos.

### CNN classifier

We use ResNet18-3D as a baseline.

Suppose you are under `hw3` directory. Train ResNet18-3D by:

```
$ python train.py
```

Run Model Inference on the test set:

```
$ python test2csv.py
```


### Submission to Kaggle

You can then submit the test outputs to the leaderboard:

```
https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw3-spring-2024
```

We use accuracy as the evaluation metric. Please refer to `test_for_student.csv` for submission format.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Try to extract different frames(other than 16) of each video with ffmpeg.
+ Try diffirent Neural Network Design.



