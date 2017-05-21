import numpy as np
from core.net_error_evaluator import NetIsNotCalculated


class NetState:
    def __init__(self):
        self.config = None
        self.net = None
        self.deviation = None
        self.is_calculated = False
        self.is_corrected = False

    def get_output_vector(self):

        if not self.is_calculated:
            raise NetIsNotCalculated

        return self.net[-1]['o']

    def clear_for_output(self):
        for l in self.net:
            del l['v']
            del l['o']
            del l['s']
