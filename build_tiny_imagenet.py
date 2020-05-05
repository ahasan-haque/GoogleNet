from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os
from lib.io import HDF5DatasetWriter

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [path.split(os.path.sep)[-3] for path in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)


trainPaths, testPaths, trainLabels, testLabels = train_test_split(trainPaths, trainLabels, 
        test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)

M = [i.split('\t')[:2] for i in 
         open(config.VAL_MAPPINGS).read().strip().split("\n")]

valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.fit_transform([m[1] for m in M])

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

# initialize the lists of RGB channel averages
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", 
            progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over the image paths

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image from disk
        image = cv2.imread(path)
        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()
