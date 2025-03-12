# Instructions for hw2

In this homework we will perform a video classification task using CNN.

## Data and Labels

Please download data from kaggle withi this [https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-2-spring-2025/data](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-2-spring-2025/data).

## Step-by-step baseline instructions

We give you a video folder, video_frames_30fpv_320p. Under each sub folder, there are 30 images from one video, we use the middle image for classification.

### Rules

You can use any or all of the 30 images to train and test your model. **You are allowed to use pretrained models and weights (only ImageNet-1K pertaining is allowed).** See this [post](https://hkust-gz.instructure.com/courses/1795/discussion_topics/12025).

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

[https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-2-spring-2025](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-2-spring-2025)

We use accuracy as the evaluation metric. Please refer to `**test_for_student.csv**` for submission format.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Try different data augmentation methods.
+ Try different split ratio of train and validation.
+ Try diffirent Neural Network Design.
+ Try diffirent overfitting methods(BN, dropout, Early Stopping).
+ Try different fusion or model aggregation methods. For example, you can simply average two model predictions (late fusion).

## Rules for Using Predefined Model APIs and Pretrained Weights

Regarding the purpose of HW2 is to allow all students in this course to get a basic understanding of how to use neural network to do prediction. So we though encourage, but are not requiring students to design neural network model architecture individually. Therefore, for the model architecture implementation, we allow students to (1) design model architectures by yourselves, (2) refer to published research works and reimplement the model architectures they proposed, or (3) use directly the predefined model APIs provided by pytorch. Note that regardless which way you implement your models, you should still keep the implementation inside the **models.py** file and follows similar coding conventions.

As for using pretrained weights, we also allow students to train your model from pretrained weights in order to get a better performance. If you intend to use pretrained weights, we would suggest you to use those pretrained on imagenet-1k. Note that whether or not you are training from pretrained weights, you should follow the class labels given from the dataset, instead of those from imagenet-1k or other datasets.

Here is an official pytorch reference in case you are interested you know how to use predefined model APIs and pretrained weights for HW2:

* [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

## Acknowledgement

Dataset borrowed from [https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw2-Spring-2024/](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw2-Spring-2024/).