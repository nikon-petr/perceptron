from math import e
from math import pow


def sigmoid(s, **kwargs):
    return 1 / (1 + e ** (-(kwargs['alpha'] * s)))


def d_sigmoid(s=None,  **kwargs):
    if kwargs.get('sigmoid') is not None:
        return kwargs['alpha'] * kwargs['sigmoid'] * (1 - kwargs['sigmoid'])
    return kwargs['alpha'] * sigmoid(s, alpha=kwargs['alpha']) * (1 - sigmoid(s, alpha=kwargs['alpha']))
