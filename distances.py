import numpy as np
import pandas as pd
import os
from bhatta_dist import bhatta_dist
from scipy.stats import norm
import scipy.integrate as integrate

def remove_allzero_features(features, n_classes):
    features = features.reshape(features.shape[0] * features.shape[1], -1)
    features = features[:, ~np.all(features == 0, axis=0)]
    features = features.reshape((n_classes, features.shape[0] // n_classes, -1))
    return features

def calculate_bhatta_hellinger_mio(features):
    n_classes = features.shape[0]
    n_samples = features.shape[1]
    n_features = features.shape[2]
    battha_distances = np.zeros((n_classes, n_classes, n_features))
    hellinger_distances = np.zeros((n_classes, n_classes, n_features))
    for i in range(n_classes):
        print('[%3d / %3d]' % (i, n_classes))
        for j in range(n_classes):
            if j <= i:
                battha_distances[i, j] = battha_distances[j, i]
                hellinger_distances[i, j] = hellinger_distances[j, i]
                continue
            for k in range(n_features):  # iterate over each feature
                if k % 50 == 0:
                    print('[%3d / %3d] features %4d' % (i, n_classes, k+1))
                distr_p = norm(np.mean(features[i, :, k]), np.std(features[i, :, k]))
                distr_q = norm(np.mean(features[j, :, k]), np.std(features[j, :, k]))
                #bc = np.sum(np.sqrt(p * q))
                bc = integrate.quad(lambda x: np.sqrt(distr_p.pdf(x) * distr_q.pdf(x)), -np.inf, np.inf)[0]
                eps = 1e-8
                battha_distances[i, j, k] = -np.log(bc)
                hellinger_distances[i, j, k] = np.sqrt(1 - bc)
                print(k, bc, battha_distances[i, j, k], hellinger_distances[i, j, k])

    return battha_distances, hellinger_distances


def calculate_bhatta(features):
# ---- BATTHA DISTANCE AMONG CENTROIDS
    n_classes = features.shape[0]
    n_samples = features.shape[1]
    n_features = features.shape[2]
    battha_distances = np.zeros((n_classes, n_classes, n_features))
    for i in range(n_classes):
        print('%3d / %3d' % (i, n_classes), end=' | ')
        for j in range(n_classes):
            if i > j:
                # distance matrix must be symmetric
                battha_distances[i, j] = battha_distances[j, i]
                continue
            for k in range(n_features):  # iterate over each feature
                if j < i:
                    battha_distances[i, j, k] = battha_distances[j, i, k]  # the resulting matrix must be symmetric
                    continue
                battha_distances[i, j, k] = bhatta_dist(features[i, :, k], features[j, :, k])
        print(battha_distances[i])

    '''
    battha_distance = n_classes x n_classes x n_features
    '''
    return battha_distances

def save_to_csv(dist_matrix, classes_names, experiment_name, distance_name, save_dir='.'):
    dist_matrix_df = pd.DataFrame(dist_matrix, index=classes_names, columns=classes_names)
    dist_matrix_df.to_csv(os.path.join(save_dir, experiment_name, distance_name+'_distances.csv'))
    print('%s file saved to %s' % (distance_name, os.path.join(save_dir, experiment_name)))
