import pprint
from copy import deepcopy
from itertools import starmap

from core.net_transfer_functions import sigmoid


class NetCalculator:
    def __init__(self, transfer_function=sigmoid, function_parameters={'alpha': 1}):
        self.__transfer_function = transfer_function
        self.__function_parameters = function_parameters

    def calculate(self, net_object, input_vector, training=False):

        if net_object.config[0] != len(input_vector):
            raise IncorrectInputVectorLength()

        calculating_net = net_object

        input_vector.insert(0, 1)

        calculating_net.layers.insert(0, [{'output': field, 'd_output': field} for field in input_vector])

        for previous_layer, current_layer in zip(calculating_net.layers, calculating_net.layers[1:-1]):
            current_layer[0]['output'] = 1
            for neuron in current_layer[1:]:
                synapse_sum = sum(starmap(lambda x, y: x['output']*y, zip(previous_layer, neuron['synapses'])))
                neuron['input'] = synapse_sum
                neuron['output'] = self.__transfer_function(synapse_sum, **self.__function_parameters)

        for neuron in calculating_net.layers[-1]:
            synapse_sum = sum(starmap(lambda x, y: x['output']*y, zip(previous_layer, neuron['synapses'])))
            neuron['input'] = synapse_sum
            neuron['output'] = self.__transfer_function(synapse_sum, **self.__function_parameters)

        calculating_net.is_calculated = True

        if not training:
            del calculating_net.layers[0]

        del input_vector[0]

        return calculating_net


class IncorrectInputVectorLength(Exception):
    pass
