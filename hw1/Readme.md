# Readme

# Instructions for hw1

In this homework we will perform a video classification task using audio-only features.

## Data and Labels

Please download data from kaggle with [this link](https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw1/data).

## Step-by-step baseline instructions

For the baselines, we will provide code and instructions for two feature representations (MFCC-Bag-of-Features and [SoundNet-Global-Pool](https://arxiv.org/pdf/1610.09001.pdf)) and three classifiers (LR, SVM and MLP). Assuming you are under Ubuntu 18.04 system and under this directory (11775-hws/spring2021/hw1/).

First, unzip mfcc.tgz by:

```
$ tar zxvf  mfcc.tgz
```

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

We already gave you the files.

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

Now we have the codebook, we will get bag-of-features (a.k.a. bag-of-words) using the codebook and the MFCCs. First, we need to get video names, we give you this videos.name.lst.


Now we extract the feature representations for each video (this would take about 7 minutes):

```
$ python get_bof.py kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

Now you can follow [here](#svm-classifier) to train SVM classifiers or [MLP](#mlp-classifier) ones.

### SoundNet-Global-Pool

Just as the MFCC-Bag-Of-Feature, we could also use the [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) model to extract a vector feature representation for each video. Since SoundNet is trained on a large dataset, this feature is usually better compared to MFCCs.

Please follow [this Github repo](https://github.com/eborboihuc/SoundNet-tensorflow) to extract audio features. Please read the paper and think about what layer(s) to use. If you save the feature representations in the same format as in the `bof/` folder, you can directly train SVM and MLP using the following instructions.

### SVM classifier

From the previous sections, we have extracted two fixed-length vector feature representations for each video. We will use them separately to train classifiers.

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

You can then submit the test outputs to the leaderboard:

```
https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw1/
```

We use accuracy as the evaluation metric. Please refer to `sample_submission.csv` for submission format.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Split `trainval.csv` into `train.csv` and `val.csv` to validate your model variants. This is important since the leaderboard limits the number of times you can submit, which means you cannot test most of your experiments on the official test set.
+ Try different number of K-Means clusters
+ Try different layers of SoundNet
+ Try out other audio features such as [VGGish Network](https://github.com/harritaylor/torchvggish) or [VGGSound](https://github.com/hche11/VGGSound)
+ Try different classifiers (different SVM kernels, different MLP hidden sizes, etc.). Please refer to [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) documentation.
+ Try different fusion or model aggregation methods. For example, you can simply average two model predictions (late fusion).

