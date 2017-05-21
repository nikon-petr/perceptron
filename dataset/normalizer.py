import numpy as np


def calculate_normalize(dataset):
    sigmas = []
    means = []
    for i in range(len(dataset[0][0])):
        means.append(np.mean(dataset[0][:,i]))
        sigmas.append(np.std(dataset[0][:,i]))
    return sigmas, means


def normalize_vector(vector, net_object):
    for i in range(len(vector)):
        vector[i] = (vector[i] - net_object.get_deviation()[0][i]) / net_object.get_deviation()[1][i]


def normalize_dataset(dataset, net_object):
    for d in range(len(dataset[0])):
        normalize_vector(dataset[0][d], net_object)
