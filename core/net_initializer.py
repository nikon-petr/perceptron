import random

from copy import deepcopy


class NetInitializer:
    def initialize(self, net_object):
        if net_object.config is None:
            raise NetConfigIndefined()

        initializing_net = deepcopy(net_object)
        config_layers = initializing_net.config

        initializing_net.layers = [[{'output': None,
                                     'synapses': [random.uniform(-1, 1)
                                     for synapse in range(previous_layer)]}
                                    for neuron in range(current_layer)]
                                   for previous_layer, current_layer in zip(config_layers, config_layers[1:])]

        return initializing_net


class NetConfigIndefined(Exception):
    pass


class NetConfigIncorrect(Exception):
    pass
