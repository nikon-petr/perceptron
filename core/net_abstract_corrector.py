from abc import ABCMeta, abstractmethod

import numpy as np

from core.net_errors import NetIsNotInitialized, NetIsNotCalculated


class Corrector:
    __metaclass__ = ABCMeta

    def __init__(self, nu):
        self.nu = nu

    @abstractmethod
    def initialize(self, net_object):
        if net_object.net[-1].get('s') is None:
            for l in range(1, len(net_object.net)):
                net_object.net[l]['s'] = np.zeros((net_object.config[l]))

    @abstractmethod
    def correct(self, net_object, output_vector):
        if net_object.net is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        self.initialize(net_object)
