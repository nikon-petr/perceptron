import numpy as np

from core.net_abstract_corrector import Corrector


class SGD(Corrector):
    def __init__(self, nu=0.1):
        super(SGD, self).__init__(nu)

    def initialize(self, net_object):
        super(SGD, self).initialize(net_object)

    def correct(self, net_object, output_vector):
        super(SGD, self).correct(net_object, output_vector)

        net = net_object.net

        df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[-1]['o'])
        net[-1]['s'] = (net[-1]['o'] - output_vector) * df

        for l in reversed(range(1, len(net))):
            g = self.nu * np.dot(net[l]['s'][:, None], np.insert(net[l - 1]['o'], 0, 1)[None, :])

            ws = np.dot(net[l]['w'][:, 1:].transpose(), net[l]['s'])
            df = np.vectorize(lambda x: net_object.df(x, **net_object.f_param))(net[l - 1]['o'])

            net[l]['w'] -= g
            net[l - 1]['s'] = ws * df

        del net[0]
