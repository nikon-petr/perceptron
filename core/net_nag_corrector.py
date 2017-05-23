import numpy as np

from core.net_abstract_corrector import Corrector
from core.net_error_evaluator import NetIsNotInitialized, NetIsNotCalculated


class NAG(Corrector):
    def __init__(self, nu=0.1, mu=0.9):
        super(NAG, self).__init__(nu)
        self.__mu = mu

    def correct(self, net_object, output_vector):
        super(NAG, self).correct(net_object, output_vector)

        net = net_object.net

        for l in net[1:]:
            l['w'] -= l['v']

        for l in range(1, len(net)):
            bo = np.insert(net[l - 1]['o'], 0, 1)
            i = np.dot(net[l]['w'], bo)
            net[l]['o'] = np.vectorize(lambda x: net_object.f(x, **net_object.f_param))(i)

        df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[-1]['o'])
        net[-1]['s'] = (net[-1]['o'] - output_vector) * df

        for l in reversed(range(1, len(net))):
            g = (1 - self.__mu) * self.nu * np.dot(net[l]['s'][:, None], np.insert(net[l - 1]['o'], 0, 1)[None, :])

            ws = np.dot(net[l]['w'][:, 1:].transpose(), net[l]['s'])
            df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[l - 1]['o'])

            net[l]['w'] -= g
            net[l]['v'] = net[l]['v'] * self.__mu + g
            net[l - 1]['s'] = ws * df

        del net[0]
