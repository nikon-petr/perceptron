from math import sqrt

import numpy as np

from core.net_abstract_corrector import Corrector


class Adam(Corrector):
    def __init__(self, nu=0.1, beta1=0.9, beta2=0.999, e=10 ** -8):
        super(Adam, self).__init__(nu)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e
        self.__t = 0

    def initialize(self, net_object):
        super(Adam, self).initialize(net_object)
        if net_object.net[-1].get('m') is None or net_object.net[-1].get('v') is None:
            for l in range(1, len(net_object.net)):
                net_object.net[l]['v'] = np.zeros((net_object.config[l], net_object.config[l - 1] + 1))
                net_object.net[l]['m'] = np.zeros((net_object.config[l], net_object.config[l - 1] + 1))

    def correct(self, net_object, output_vector):
        super(Adam, self).correct(net_object, output_vector)
        self.__t += 1

        net = net_object.net

        df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[-1]['o'])
        net[-1]['s'] = (net[-1]['o'] - output_vector) * df

        for l in reversed(range(1, len(net))):
            g = np.dot(net[l]['s'][:, None], np.insert(net[l - 1]['o'], 0, 1)[None, :])

            ws = np.dot(net[l]['w'][:, 1:].transpose(), net[l]['s'])
            df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[l - 1]['o'])

            net[l]['m'] = net[l]['m'] * self.beta1 + (1 - self.beta1) * g
            net[l]['v'] = net[l]['v'] * self.beta2 + (1 - self.beta2) * g ** 2

            m = net[l]['m'] / (1 - self.beta1 ** self.__t)
            v = net[l]['v'] / (1 - self.beta2 ** self.__t)

            net[l]['w'] -= self.nu * m / np.vectorize(lambda x: sqrt(x))(v + self.e)
            net[l - 1]['s'] = ws * df

        del net[0]
