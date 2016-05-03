# Ian London 2016
# tools to run a Visual Bag of Words classifier on any images

import cv2
import numpy as np
import glob
import os
from sklearn.cluster import KMeans

print 'OpenCV VERSION (should be 3.1.0 or later, with nonfree modules installed!):', cv2.__version__

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError("Unable to open '%s'. Are you sure it's a valid image path?")
    return img

def neg_img_cal101(positive_folder, cal101_root='101_ObjectCategories', image_suffix='*.jpg'):
    """Simply return list of paths for all images in cal101 dataset, except those in positive_folder."""
    return [path for path in glob.glob(cal101_root + '/*/' + image_suffix) if positive_folder not in path]

def binary_labeled_img_from_cal101(positive_folder, cal101_root='101_ObjectCategories', image_suffix='*.jpg'):
    """
    Generate a balanced dataset of positive and negative images from a directory of images
    where each type of image is separated in its own folder.

    Returns:
    --------
    labeled_img_paths: list of lists
        Of the form [[image_path, label], ...]
        Where label is True or False for positive and negative images respectively
    """
    all_imgs = set(glob.glob(cal101_root + '/*/' + image_suffix))
    pos_imgs = set(glob.glob(os.path.join(cal101_root, positive_folder) + '/' + image_suffix))
    neg_imgs = all_imgs - pos_imgs

    neg_sample_size = len(pos_imgs)
    selected_negs = np.random.choice(list(neg_imgs), size=neg_sample_size, replace=False)

    print '%i positive, %i negative images selected (out of %i negatives total)' % (
        len(pos_imgs), len(selected_negs), len(neg_imgs))

    labeled_img_paths = [[path, True] for path in pos_imgs] + [[path, False] for path in selected_negs]

    return np.array(labeled_img_paths)


def train_test_split_idxs(total_rows, test_train_ratio):
    """
    Get indexes for training and test rows, given a total number of rows.
    Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]

    Returns:
    --------
    training_idxs, test_idxs
        Both lists of integers
    """
    row_range = range(total_rows)
    no_training_rows = int(total_rows*(1-test_train_ratio))
    training_idxs = np.random.choice(row_range, size=no_training_rows, replace=False)
    test_idxs = np.array(list(set(row_range) - set(training_idxs)))

    print 'Train-test split: %i training rows, %i test rows' % (len(training_idxs), len(test_idxs))

    return training_idxs, test_idxs


def gen_bow_features(labeled_img_paths, test_train_ratio, K_clusters):
    """
    Generate "visual bag of words" features for a set of images.

    Parameters:
    -----------
    labeled_img_paths: list of lists
        Of the form [[image_path, label], ...]
        The label should identify the image type. Eg 'schooner', or 18, (or even True/False for a binary detector)
        The image_path should the full absolute path name as a string

    test_train_ratio : float
        Percentage of test rows to training rows.

    K_clusters : int
        The 'K' in KMeans. This equals the number of final "visual words" you get, which
        also equals the number of feature columns in your final X matrices.

    Returns:
    --------
    X_train, X_test, y_train, y_test, kmeans :
        X's have K feature columns, each column corresponding to a visual word
    """
    if not 0 < test_train_ratio < 1:
        raise ValueError('test_train_ratio must be between 0 and 1 (non-inclusive)')

    # img_keypoints = {}
    img_descs = []

    print 'generating SIFT keypoints for %i images' % len(labeled_img_paths)

    for img_path, label in labeled_img_paths:
        img = read_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray, None)
        # img_keypoints[img_path] = kp
        img_descs.append(desc)

    # Generate indexes of training rows
    total_rows = len(img_descs)
    training_idxs, test_idxs = train_test_split_idxs(total_rows, test_train_ratio)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_descriptors = np.array(all_descriptors)

    assert all_descriptors.shape[1] == 128
    print '%i descriptors before clustering' % all_descriptors.shape[0]


    # Cluster descriptors to get codebook
    print 'Clustering on training set to get codebook of %i words' % K_clusters
    kmeans = KMeans(n_clusters=K_clusters, n_jobs=-1)
    # train kmeans on those descriptors selected above
    kmeans.fit(all_descriptors)
    print 'done clustering'

    # compute set of cluster-reduced words for each image
    img_clustered_words = [kmeans.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=K_clusters) for clustered_words in img_clustered_words])

    X_train = img_bow_hist[training_idxs]
    X_test = img_bow_hist[test_idxs]

    y_labels = np.array(labeled_img_paths)[:,1]
    y_train = y_labels[training_idxs]
    y_test = y_labels[test_idxs]

    return X_train, X_test, y_train, y_test, kmeans

def img_to_vect(img_path, kmeans):
    """
    Given an image path and a trained kmeans classifier,
    generates a feature vector representing that image
    """

    img = read_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = kmeans.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=kmeans.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)
