from copy import deepcopy
from itertools import starmap

from core.net_transfer_functions import d_sigmoid
from core.net_error_evaluator import NetIsNotInitialized, NetIsNotCalculated


class CM:
    def __init__(
            self,
            nu=0.2,
            mu=0.9,
            d_transfer_function=d_sigmoid,
            function_parameters={'alpha': 1}):

        self.__nu = nu
        self.__mu = mu
        self.__df = d_transfer_function
        self.__df_param = function_parameters

    def correct(self, net_object, expected_output_vector):

        if net_object.layers is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        correcting_net = deepcopy(net_object)

        for previous_layer, current_layer in zip(correcting_net.layers, correcting_net.layers[1:]):
            for neuron in current_layer:
                synapse_sum = sum(starmap(lambda x, y: x['d_output']*y, zip(previous_layer, neuron['synapses'])))
                neuron['d_input'] = synapse_sum
                neuron['d_output'] = self.__df(sigmoid=neuron['output'], **self.__df_param)

        for neuron, d in zip(correcting_net.layers[-1], expected_output_vector):
            neuron['sigma'] = (neuron['d_output'] - d) * self.__df(neuron['d_input'], **self.__df_param)

        for layer_n in range(len(correcting_net.layers)-2, -1, -1):
            for neuron_n in range(len(correcting_net.layers[layer_n])):
                sigma = sum([n['sigma'] * n['synapses'][neuron_n] for n in correcting_net.layers[layer_n+1]])
                sigma *= correcting_net.layers[layer_n][neuron_n]['d_output']
                correcting_net.layers[layer_n][neuron_n]['sigma'] = sigma

        for layer_n in range(1, len(correcting_net.layers)):
            for neuron_n in range(len(correcting_net.layers[layer_n])):

                if not correcting_net.layers[layer_n][neuron_n].get('velocity'):
                    velocity = [0 for synapse in range(len(correcting_net.layers[layer_n-1]))]
                    correcting_net.layers[layer_n][neuron_n]['velocity'] = velocity

                for synapse_n in range(len(correcting_net.layers[layer_n][neuron_n]['synapses'])):

                    sigma = correcting_net.layers[layer_n][neuron_n]['sigma']
                    signal = correcting_net.layers[layer_n-1][synapse_n]['output']

                    velocity = correcting_net.layers[layer_n][neuron_n]['velocity'][synapse_n]
                    velocity = self.__mu * velocity + (1 - self.__mu) * sigma * signal
                    correcting_net.layers[layer_n][neuron_n]['velocity'][synapse_n] = velocity

                    correcting_net.layers[layer_n][neuron_n]['synapses'][synapse_n] -= velocity

        del correcting_net.layers[0]

        correcting_net.is_corrected = True
        correcting_net.is_calculated = False

        return correcting_net
