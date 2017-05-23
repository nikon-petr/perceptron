import numpy as np


def calculate_normalize(dataset):
    deviations = []
    means = []
    for i in range(len(dataset[0][0])):
        means.append(np.mean(dataset[0][:,i]))
        deviations.append(np.std(dataset[0][:,i]))
    return {'means': means, 'deviations': deviations}


def normalize_vector(vector, net_object):
    for i in range(len(vector)):
        vector[i] = (vector[i] - net_object.get_normalization()['means'][i]) / net_object.get_normalization()['deviations'][i]


def normalize_dataset(dataset, net_object):
    for d in range(len(dataset[0])):
        normalize_vector(dataset[0][d], net_object)
