#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster import KMeans
import sys
import time
import collections
import csv
import argparse
from tqdm import tqdm
# Generate MFCC-Bag-of-Word features for videos
# each video is represented by a single vector

parser = argparse.ArgumentParser()
parser.add_argument("kmeans_model")
parser.add_argument("cluster_num", type=int)
parser.add_argument("file_list")
parser.add_argument("--mfcc_path", default="mfcc")
parser.add_argument("--output_path", default="bof")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. load the kmeans model
  kmeans = pickle.load(open(args.kmeans_model, "rb"))

  # 2. iterate over each video and
  # use kmeans.predict(mfcc_features_of_video)
  start = time.time()
  fread = open(args.file_list, "r")
  for line in tqdm(fread.readlines()):
    mfcc_path = os.path.join(args.mfcc_path, line.strip() + ".mfcc.csv")
    bof_path = os.path.join(args.output_path, line.strip() + ".csv")

    if not os.path.exists(mfcc_path):
      continue
    # (num_frames, d)
    array = numpy.genfromtxt(mfcc_path, delimiter=";")
    # (num_frames,), each row is an integer for the clostest cluster center

    # create dict containing 0 count for cluster number
 
    # {0: count_for_0, 1: count_for_1, ...}

    # normalize the frequency by dividing with frame number


    numpy.savetxt(bof_path, list_freq)

  end = time.time()
  print("K-means features generated successfully!")
  print("Time for computation: ", (end - start))