from math import sqrt


def mean(dataset, index):
    """calculates mean"""
    sum = 0
    for (inv, outv) in dataset:
        sum += inv[index]
    return sum / len(dataset)


def stddev(dataset, index):
    """calculates standard deviation"""
    sum = 0
    mn = mean(dataset, index)
    for (inv, outv) in dataset:
        sum += (inv[index] - mn) ** 2
    return sqrt(sum / len(dataset))


def normalize(dataset):
    sigmas = []
    means = []
    for i in range(len(dataset[0][0])):
        means.append(mean(dataset, i))
        sigmas.append(stddev(dataset, i))
        for (inv, outv) in dataset:
            inv[i] = (inv[i] - means[i]) / sigmas[i]
    return sigmas, means


def normalize_vector(vector, net_object):
    for i in range(len(vector)):
        vector[i] = (vector[i] - net_object.get_diviation()[0][i]) / net_object.get_diviation()[1][i]