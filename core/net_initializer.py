import numpy as np

from core.net_errors import NetConfigIndefined, IncorrectFactorValue


def initialize(net_object, factor=0.01):
    if net_object.config is None:
        raise NetConfigIndefined()
    if abs(factor) > 1:
        raise IncorrectFactorValue()

    net_object.net = []

    for l in range(1, len(net_object.config)):
        net_object.net.append({
            'w': np.random.uniform(-factor, factor, (net_object.config[l], net_object.config[l - 1] + 1)),
            'v': np.zeros((net_object.config[l], net_object.config[l - 1] + 1)),
            'o': np.zeros((net_object.config[l])),
            's': np.zeros((net_object.config[l])),
        })
