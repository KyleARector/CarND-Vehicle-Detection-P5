import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import pickle
from lesson_functions import *


# Function to train the classifier against the images of cars and non
# cars. Returns X_Scaler and the model (svc)
def train_classifier(cars, notcars, color_space="RGB", orient=9,
                     pix_per_cell=2, cell_per_block=2, hog_channel=0):
    t = time.time()
    car_features = extract_features(cars,
                                    color_space=color_space,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
    notcar_features = extract_features(notcars,
                                       color_space=color_space,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), "Seconds to extract HOG features...")
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print("Using:", orient, "orientations", pix_per_cell,
          "pixels per cell and", cell_per_block, "cells per block")
    print("Feature vector length:", len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), "Seconds to train SVC...")
    # Check the score of the SVC
    print("Test Accuracy of SVC = ", round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print("My SVC predicts: ", svc.predict(X_test[0:n_predict]))
    print("For these", n_predict, "labels: ", y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), "Seconds to predict", n_predict, "labels with SVC")
    return X_scaler, svc


def main():
    # Tuning parameters
    color_space = "YCrCb"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

    # Divide up into cars and notcars
    cars = glob.glob("../../vehicle_training/cars/*")
    notcars = glob.glob("../../vehicle_training/not_cars/*")

    X_scaler, svc = train_classifier(cars,
                                     notcars,
                                     color_space=color_space,
                                     orient=orient,
                                     pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel)

    pickle.dump({"X_scaler": X_scaler, "svc": svc},
                open("trained_model.p", "wb"))


if __name__ == "__main__":
    main()
