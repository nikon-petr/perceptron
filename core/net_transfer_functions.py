from math import e
from math import pow


def linear(k, s):
    return k * s


def sigmoid(s, alpha):
    return 1 / (1 + pow(e, - alpha * s))


def d_sigmoid(s, alpha):
    return alpha * sigmoid(alpha, s) * (1 - sigmoid(alpha, s))
