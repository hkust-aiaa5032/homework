#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time
import collections
import csv
import pdb
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("kmeans_model -- path to the kmeans model")
        print("cluster_num -- number of cluster")
        print("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # 1. Set output path for each video representation
    #output = '../kmeans'

    # 2. load the kmeans model
    kmeans = pickle.load(open(kmeans_model, "rb"))

    # 3. iterate over each video and use kmeans.predict(mfcc_features_of_video) and save the results in this format video_name.kmeans
    start = time.time()
    fread = open(file_list, "r")
    for line in fread.readlines():
        kmeans_path = 'kmeans/' + line.replace('\n', '') + '.kmeans.csv'
        mfcc_path = 'mfcc/' + line.replace('\n', '') + '.mfcc.csv'

        if(os.path.exists(mfcc_path) is False):
            continue
        array = numpy.genfromtxt(mfcc_path, delimiter=';')
        kmeans_result = kmeans.predict(array)

        dict_freq = collections.Counter(kmeans_result)
        # create dict containing 0 count for cluster number
        keys = numpy.arange(0, cluster_num, 1)
        #values = numpy.zeros(cluster_num, dtype='int')
        values = numpy.zeros(cluster_num, dtype='float')
        dict2 = dict(zip(keys, values))
        dict2.update(dict_freq)
        list_freq = list(dict2.values())
        # normalize the frequency by dividing with frame number
        # pdb.set_trace()
        list_freq = numpy.array(list_freq) / array.shape[0]
        numpy.savetxt(kmeans_path, list_freq)

    end = time.time()
    print("K-means features generated successfully!")
    print("Time for computation: ", (end - start))
