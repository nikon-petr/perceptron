import pprint
import numpy as np


def initialize(net_object, factor=0.01):
    if net_object.config is None:
        raise NetConfigIndefined()

    net_object.net = []

    for l in range(1, len(net_object.config)):
        net_object.net.append({
            'w': np.random.uniform(-factor, factor, (net_object.config[l], net_object.config[l-1] + 1)),
            'v': np.zeros((net_object.config[l], net_object.config[l-1] + 1)),
            'o': np.zeros((net_object.config[l])),
            's': np.zeros((net_object.config[l])),
        })


class NetConfigIndefined(Exception):
    pass


class NetConfigIncorrect(Exception):
    pass
