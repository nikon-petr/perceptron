import numpy as np


def evaluate(net_object, expected_output_vector):
    if net_object.net is None:
        raise NetIsNotInitialized()
    if not net_object.is_calculated:
        raise NetIsNotCalculated()

    e = np.mean(np.vectorize(lambda x1, x2: (x1 - x2) ** 2)(net_object.net[-1]['o'], expected_output_vector))

    return e


class NetIsNotInitialized(Exception):
    pass


class NetIsNotCalculated(Exception):
    pass


class IncorrectExpectedOutputVectorLength(Exception):
    pass
