from copy import deepcopy
from itertools import starmap


class NetCalculator:
    def __init__(self, transfer_function, function_parameters):
        self.__transfer_function = transfer_function
        self.__function_parameters = function_parameters

    def calculate(self, net_object, input_vector):

        if net_object.config[0] != len(input_vector):
            raise IncorrectInputVectorLength()

        calculating_net = deepcopy(net_object)

        calculating_net.layers.insert(0, [{'output': field, 'd_output': field} for field in input_vector])

        for previous_layer, current_layer in zip(calculating_net.layers, calculating_net.layers[1:]):
            for neuron in current_layer:
                synapse_sum = sum(starmap(lambda x, y: x['output']*y, zip(previous_layer, neuron['synapses'])))
                neuron['input'] = synapse_sum
                neuron['output'] = self.__transfer_function(synapse_sum, **self.__function_parameters)

        calculating_net.is_calculated = True

        return calculating_net


class IncorrectInputVectorLength(Exception):
    pass
