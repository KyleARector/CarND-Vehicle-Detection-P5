import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.cross_validation import train_test_split
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import os


# Global variables
svc, X_scaler = [], []

# Tuning parameters
color_space = "YCrCb"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 16
ystart = 400
ystop = 780


# Function to find window segments of an image that have car features
def search_windows(img, windows, clf, scaler, color_space="RGB",
                   spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64))
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
              cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv="RGB2YCrCb")
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # 2 Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell,
                            cell_per_block, feature_vec=False)

    hot_windows = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,
                             xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window,
                                                xleft:xleft+window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            feats = np.hstack((spatial_features,
                               hist_features,
                               hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(feats)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw, ytop_draw+win_draw+ystart),
                              (0, 0, 255), 6)
                hot_windows.append(((xbox_left, ytop_draw+ystart),
                                    (xbox_left+win_draw,
                                     ytop_draw+win_draw+ystart)))
    return hot_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def heat_mapping(img, box_list):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    return labels


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Process all static images in given directory
def process_images(in_directory, out_directory):
    for image in os.listdir(in_directory):
        out_img = pipeline(mpimg.imread(in_directory + image))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_directory + "vehicle_detected_" + image, out_img)


# Add vehicle detection to all frames of a video
def process_video(in_video, out_video="video_out.mp4"):
    video_output = out_video
    clip1 = VideoFileClip(in_video)
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(video_output, audio=False)


# Main image processing pipeline
def pipeline(img):
    pos_scales = [1.0, 1.5]
    master_boxes = []
    for scale in pos_scales:
        boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                          orient, pix_per_cell, cell_per_block, spatial_size,
                          hist_bins)
        master_boxes = master_boxes + boxes
    labels = heat_mapping(img, master_boxes)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


def main():
    global X_scaler, svc
    in_directory = "test_images/"
    out_directory = "output_images/"

    # Load pickled training data
    trained_model = pickle.load(open("trained_model.p", "rb"))
    X_scaler, svc = map(trained_model.get, ("X_scaler", "svc"))

    # Process static images
    process_images(in_directory, out_directory)
    process_video("project_video.mp4")


if __name__ == "__main__":
    main()
