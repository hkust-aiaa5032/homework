# Instructions for hw1

In this homework we will perform a video classification task (10-class classification, 5k samples for training and 2280 samples for testing) using audio-only features.

## Data and Labels

Please download data from kaggle with [this link](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-1-spring-2026/data).
## Step-by-step baseline instructions

For the baselines, we will provide code (templates) and instructions for a feature extraction method (MFCC-Bag-of-Features) and three classifiers (LR, SVM and MLP). Assuming you are under Ubuntu OS and under this directory (homework/hw1/). Open a shell terminal to run the commands.

You will need to complete the code in `train_kmeans.py, get_bof.py, train_*.py, test_*.py` to get a baseline output.
You could follow the comments we provide or you could write your own.

### MFCC-Bag-Of-Features

Let's create the folders we need first:

```
$ mkdir bof/ labels/
```

Put trainval.csv and test_for_student.label under labels/.

1. Dependencies:  

Install python dependencies by:

```
$ pip install scikit-learn pandas tqdm numpy
```

2. Get MFCCs

You have downloaded the files from Kaggle. First, unzip mfcc.tgz by:

```
$ tar zxvf  mfcc.tgz
```

3. K-Means clustering

As taught in the class, we will use K-Means to get feature codebook from the MFCCs. Since there are too many feature lines, we will randomly select a subset (20%) for K-Means clustering by:

```
$ python select_frames.py labels/trainval.csv 0.2 selected.mfcc.csv --mfcc_path mfcc/
```

Now we train it by (50 clusters, this would take about 5 minutes):

```
$ python train_kmeans.py selected.mfcc.csv 50 kmeans.50.model
```

4. Feature extraction

Now we have the codebook, we will get bag-of-features (a.k.a. bag-of-words) using the codebook and the MFCCs. First, we need to get video names. We give you this in `videos.name.lst`.


Now we extract the feature representations for each video (this would take about 7 minutes):

```
$ python get_bof.py kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

<!-- Now you can follow [here](#svm-classifier) to train SVM classifiers or [MLP](#mlp-classifier) ones. -->
We provide the training and testing of three baselines below.

### SVM classifier

From the previous sections, we have extracted the MFCC-Bag-of-Features representation for each video. We will use this to train different classifiers.

Suppose you are under `hw1` directory. Train SVM by:

```
$ mkdir models/
$ python train_svm_multiclass.py bof/ 50 labels/trainval.csv models/mfcc-50.svm.multiclass.model
```

Run SVM on the test set:

```
$ python test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 labels/test_for_student.label mfcc-50.svm.multiclass.csv
```



### LR classifier

Suppose you are under `hw1` directory. Train LR by:

```
$ python train_LR.py bof/ 50 labels/trainval.csv models/mfcc-50.LR.model
```

Test:

```
$ python test_LR.py models/mfcc-50.LR.model bof 50 labels/test_for_student.label mfcc-50.LR.csv
```

### MLP classifier

Suppose you are under `hw1` directory. Train MLP by:

```
$ python train_mlp.py bof/ 50 labels/trainval.csv models/mfcc-50.mlp.model
```

Test:

```
$ python test_mlp.py models/mfcc-50.mlp.model bof 50 labels/test_for_student.label mfcc-50.mlp.csv
```


### Submission to Kaggle

You can then submit the test outputs (the `*.csv` files) to the [leaderboard](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-1-spring-2026/leaderboard):

We use accuracy as the evaluation metric. Please refer to `sample_submission.csv` for submission format.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Split `trainval.csv` into `train.csv` and `val.csv` to validate your model variants. This is important since the leaderboard limits the number of times you can submit, which means you cannot test most of your experiments on the official test set.
+ Try different number of K-Means clusters
+ Try different hyper-parameters for your models (different SVM kernels and Cs, different MLP hidden sizes, etc.). Please refer to [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) documentation.
+ Try different fusion or model aggregation methods. For example, you can simply average two model predictions (late fusion).
+ Try bagging and boosting

