import pprint

import numpy as np

from core.net_transfer_functions import sigmoid, tanh


def calculate(net_object, input_vector, training=False, f=tanh, p={'alpha': 1}):

    if net_object.config[0] != len(input_vector):
        raise IncorrectInputVectorLength()

    net_object.net.insert(0, {'o': input_vector})

    for l in range(1, len(net_object.net)):
        bo = np.insert(net_object.net[l-1]['o'], 0, 1)
        i = np.dot(net_object.net[l]['w'], bo)
        net_object.net[l]['o'] = np.vectorize(lambda x: f(x, **p))(i)

    if not training:
        del net_object.net[0]

    net_object.is_calculated = True


class IncorrectInputVectorLength(Exception):
    pass
