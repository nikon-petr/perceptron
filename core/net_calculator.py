import numpy as np

from core.net_errors import IncorrectInputVectorLength


def calculate(net_object, input_vector, training=False):
    if net_object.config[0] != len(input_vector):
        raise IncorrectInputVectorLength()

    net = net_object.net

    net.insert(0, {'o': input_vector})

    for l in range(1, len(net)):
        bo = np.insert(net[l - 1]['o'], 0, 1)
        i = np.dot(net[l]['w'], bo)
        net[l]['o'] = np.vectorize(lambda x: net_object.f(x, **net_object.f_param))(i)

    if not training:
        del net[0]

    net_object.is_calculated = True
