import os
import cv2
import numpy as np
import csv
from os import listdir
from descriptors import haralick, haralick_with_mean, bitdesc, glcm
from paths import kth_path, kth_dir


def read_files(path):
    '''
    Print all the files in a given path and 
    its subfolders
    '''
    for root, directories, files in os.walk(path):
        for name in files:
            print(os.path.join(root, name), ":", directories)


def read_extract(path):
    '''
    Print all the files in a given path and 
    its subfolders
    '''
    for root, directories, files in os.walk(path):
        for name in files:
            print(os.path.join(root, name))
            img = os.path.join(root, name)
            img_gray = cv2.imread(img, 0)
            haralik_feat = haralick_with_mean(img_gray)
            print(haralik_feat)


def feat_extraction():
    for kth_type in kth_dir:
        for filename in listdir(kth_path+kth_type+'/'):
            print('Class: %s -- (%s) -> Image : %s' %
                  (kth_type, kth_dir.index(kth_type), filename))


def extract_features(image, n=10):
    """
    Extract texture features from an input image.

    Parameters:
    -----------
    image: numpy array
        Input image as a numpy array.
    n: int, optional (default=10)
        Number of bit-planes to use for Bit-Plane feature extraction.

    Returns:
    --------
    features: numpy array
        Concatenated feature vector.
    """
    # Calculate Bit-Plane features
    bitplane_features = bitdesc(image, n=n)
    if bitplane_features is None:
        bitplane_features = np.zeros((n,), dtype=np.float32)
    else:
        bitplane_features = np.array(bitplane_features)

    # Calculate Haralick features
    haralick_features = np.array(haralick_with_mean(image))

    # Calculate GLCM features
    glcm_features = np.array(glcm(image))

    # Concatenate feature vectors
    if bitplane_features.size == haralick_features.size == glcm_features.size:
        features = np.hstack(
            (bitplane_features, haralick_features, glcm_features))
    else:
        print("Error: Feature vectors do not have the same size.")
        features = None

    return features


def extract_features_and_write_csv(dataset_path, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                if not os.path.isfile(image_path):
                    print(f"Erreur : {image_path} n'existe pas")
                    continue  # passer à l'image suivante si le chemin d'accès n'existe pas

                img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    print(
                        f"Erreur : Impossible de charger l'image {image_path}")
                    continue
# passer à l'image suivante si l'image n'a pas été chargée correctement

                features = extract_features(img_gray)
                row = list(features) + [folder_name]
                writer.writerow(row)


def main():
    path = "datasets/"
    # read_files(path)
    # read_extract(path)
    feat_extraction()

    dataset_path = "datasets/TIPS2-a"
    csv_file_path = "features.csv"
    extract_features_and_write_csv(dataset_path, csv_file_path)


if __name__ == '__main__':
    main()
