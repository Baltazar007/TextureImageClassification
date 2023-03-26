import cv2
import mahotas.features as ft
import numpy as np
from skimage.feature import graycomatrix, graycoprops  # scikit-image
import BiT  # Bitdesc
from BiT import biodiversity, taxonomy, validate_species_vector  # Bitdesc
# Défine Haralick


def haralick(data):  # numpay array
    # Mean(0) retourn la moyenne des array pour quelle fonctionne nous devons avoir le array en meme taille
    all_statistics = ft.haralick(data)
    return all_statistics


def haralick_with_mean(data):
    # Mean(0) retourn la moyenne des array pour quelle fonctionne nous devons avoir le array en meme taille
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics

# Define Bitdesc


def bitdesc(data):
    bio = biodiversity(data)
    taxo = taxonomy(data)
    species = None  # ajout d'une valeur par défaut
    all_statistics = bio + taxo
    if data is not None:
        species = validate_species_vector(data.flatten())
    all_statistics += species
    return all_statistics

# Gray-Leve Co-occurence Matrix


def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    dissimularite = graycoprops(glcm, 'dissimilarity')[0]
    contraste = graycoprops(glcm, 'contrast')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    energy = graycoprops(glcm, 'energy')[0]
    homogeneite = graycoprops(glcm, 'homogeneity')[0]
    ASM = graycoprops(glcm, 'ASM')[0]
    features = [dissimularite, contraste,
                correlation, energy, homogeneite, ASM]
    return features
