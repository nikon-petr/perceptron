import pprint
from copy import deepcopy
from itertools import starmap

import numpy as np

from core.net_transfer_functions import sigmoid, d_sigmoid
from core.net_error_evaluator import NetIsNotInitialized, NetIsNotCalculated


class SC:
    def __init__(self, nu=0.1, df=d_sigmoid, df_p={'alpha': 1}):
        self.nu = nu
        self.__df = df
        self.__df_p = df_p

    def correct(self, net_object, output_vector):

        if net_object.net is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        net = net_object.net

        net[-1]['s'] = (net[-1]['o'] - output_vector) * np.vectorize(lambda x: self.__df(x, **self.__df_p))(net[-1]['o'])

        for l in reversed(range(1, len(net))):
            g = self.nu * np.dot(net[l]['s'][:,None], np.insert(net[l-1]['o'], 0, 1)[None,:])

            ws = np.dot(net[l]['w'][:, 1:].transpose(), net[l]['s'])
            df = np.vectorize(lambda x: self.__df(x, **self.__df_p))(net[l - 1]['o'])

            net[l]['w'] -= g
            net[l - 1]['s'] = ws * df

        del net[0]
