#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
import numpy as np

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. load svm model
  svm = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

  X = np.array()

  # 3. Get scores with trained svm model
  # (num_samples, num_class)
  scoress = 

  # 4. save the argmax decisions for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, scores in enumerate(scoress):
      predicted_class = np.argmax(scores)
      f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
