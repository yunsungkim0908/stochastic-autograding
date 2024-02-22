import json
import numpy as np
import os
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from stochasticgrade.constants import *
from tqdm import tqdm


def agglomerative(score_list, n_clusters=20):    
    """
    Performs agglomerative hierarchical clustering, creating `n_clusters` total.
    Accepts a score_list: a list of lists of scores for various scorers.
    Returns the cluster labels.
    """
    scores = np.column_stack(score_list)
    
    print('Performing agglomerative hierarchical clustering...')    
    algorithm = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = algorithm.fit_predict(scores)
    print('Done.')
    return cluster_labels


def hdbscan(score_list):    
    """
    Performs HDBSCAN clustering.
    Accepts a score_list: a list of lists of scores for various scorers.
    Returns the cluster labels.
    """
    data = []
    for i in range(len(score_list[0])):
        example = []
        for j in range(len(score_list)):  
            score = score_list[j][i]
            example.append(score)
        data.append(np.array(example))
    scores = np.array(data)
    
    print('Performing HDBSCAN clustering...')    
    algorithm = sklearn.cluster.HDBSCAN()
    cluster_labels = algorithm.fit_predict(scores)
    print('Done.')
    return cluster_labels


def gaussian_mixture_model(score_list, n_clusters=20):    
    """
    Performs Gaussian Mixture Model (GMM) clustering, creating `n_clusters` total.
    Accepts a score_list: a list of lists of scores for various scorers.
    Returns the cluster labels.
    """
    data = []
    for i in range(len(score_list[0])):
        example = []
        for j in range(len(score_list)):  
            score = score_list[j][i]
            example.append(score)
        data.append(np.array(example))
    scores = np.array(data)
    
    print('Performing Gaussian Mixture Model clustering...')    
    algorithm = sklearn.mixture.GaussianMixture(n_components=n_clusters)
    cluster_labels = algorithm.fit_predict(scores)
    print('Done.')
    return cluster_labels


def mean_shift(scores, bandwidth=5):
    """
    Performs mean shift clustering.
    Accepts a score_list: a list of lists of scores for various scorers.
    Returns the cluster labels. 
    """
    data = []
    for i in range(len(score_list[0])):
        example = []
        for j in range(len(score_list)):
            example.append(score_list[j][i])
        data.append(np.array(example))
    scores = np.array(data)
    
    print('Performing mean shift clustering...')
    algorithm = sklearn.cluster.MeanShift(bandwidth=bandwidth)
    cluster_labels = algorithm.fit_predict(scores.reshape(-1, 1))
    print('Done.')
    return cluster_labels