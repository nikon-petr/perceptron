import numpy as np

from core.net_error_evaluator import NetIsNotInitialized, NetIsNotCalculated


class NAG:
    def __init__(self, nu=0.1, mu=0.9):

        self.__nu = nu
        self.__mu = mu

    def correct(self, net_object, output_vector):

        if net_object.net is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        net = net_object.net

        for l in net[1:]:
            l['w'] += l['v']

        for l in range(1, len(net)):
            i = np.dot(net[l]['w'], np.insert(net[l - 1]['o'], 0, 1))
            net[l]['o'] = np.vectorize(lambda x: net_object.f(x, **net_object.f_param))(i)

        net[-1]['s'] = (net[-1]['o'] - output_vector) * np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[-1]['o'])

        for l in reversed(range(1, len(net))):
            net[l - 1]['s'] = np.dot(net[l]['w'][:, 1:].transpose(), net[l]['s']) * np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[l - 1]['o'])
            delta = -self.nu * np.dot(np.insert(net[l - 1]['o'], 0, 1)[:, None], net[l]['s'][None, :]).transpose()
            net[l]['v'] = net[l]['v'] * self.__mu + delta
            net[l]['w'] += delta

        del net[0]
