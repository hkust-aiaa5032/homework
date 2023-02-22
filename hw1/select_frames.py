#!/bin/python
# Randomly select MFCC frames

import argparse
import numpy
import os
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("file_list")
parser.add_argument("select_ratio", type=float)
parser.add_argument("output_file")
parser.add_argument("--mfcc_path", default="mfcc")

if __name__ == "__main__":
  args = parser.parse_args()

  fread = open(args.file_list, "r")
  fwrite = open(args.output_file, "w")

  # random selection is done by randomizing the rows of the whole matrix, and then selecting the first
  # num_of_frame * ratio rows
  numpy.random.seed(18877)

  for line in tqdm(fread.readlines()[1:]):  # skipped the header
    mfcc_path = os.path.join(args.mfcc_path, line.strip().split(",")[0] + ".mfcc.csv")
    if not os.path.exists(mfcc_path):
      continue
    array = numpy.genfromtxt(mfcc_path, delimiter=";")
    numpy.random.shuffle(array)
    select_size = int(array.shape[0] * args.select_ratio)
    feat_dim = array.shape[1]

    for n in range(select_size):
      line = str(array[n][0])
      for m in range(1, feat_dim):
        line += ";" + str(array[n][m])
      fwrite.write(line + "\n")
  fwrite.close()

