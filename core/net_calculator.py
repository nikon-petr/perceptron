import pprint

from copy import deepcopy
from itertools import accumulate, starmap


class NetCalculator:
    def __init__(self, transfer_function):
        self.__transfer_function = transfer_function

    def calculate(self, net_object, input_vector):

        if net_object.config[0] != len(input_vector):
            raise IncorrectInputVectorLength()

        calculating_net = deepcopy(net_object)

        calculating_net.layers.insert(0, [{'output': field} for field in input_vector])

        for previous_layer, current_layer in zip(calculating_net.layers, calculating_net.layers[1:]):
            for neuron in current_layer:
                synapse_sum = sum(starmap(lambda x, y: x['output']*y, zip(previous_layer, neuron['synapses'])))
                neuron['output'] = self.__transfer_function(synapse_sum, 1)

        pprint.pprint(calculating_net.layers, indent=4)


class IncorrectInputVectorLength(Exception):
    pass
