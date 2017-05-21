import os
import pprint
from copy import deepcopy

import numpy as np

from json import load
from json import dump
from json import JSONDecodeError


def upload(net_object, path):
    if not os.path.isfile(path):
        raise JsonFileNotFound()

    try:
        with open(path, 'r') as file:
            deserialized_file = load(file)
            net_object.config = deserialized_file['config']
            net_object.net = deserialized_file.get('net')
            net_object.deviation = deserialized_file.get('deviation')

            if net_object.net:
                for l in range(1, len(net_object.config)):
                    net_object.net[l-1]['w'] = np.array(net_object.net[l-1]['w'])
                    net_object.net[l-1]['v'] = np.zeros((net_object.config[l], net_object.config[l-1]+1))
                    net_object.net[l-1]['o'] = np.zeros((net_object.config[l]))
                    net_object.net[l-1]['s'] = np.zeros((net_object.config[l]))

    except KeyError:
        raise JsonFileStructureIncorrect()
    except JSONDecodeError:
        raise


def unload(net_object, path):
    try:
        net_copy = deepcopy(net_object.net)
        for l in net_copy:
            l['w'] = l['w'].tolist()
            del l['v']
            del l['o']
            del l['s']

        with open(path, 'w') as file:
            file_dictionary = {
                'config': net_object.config,
                'net': net_copy,
                'deviation': net_object.deviation
            }
            dump(file_dictionary, file, sort_keys=True, indent=4)
    except JSONDecodeError:
        raise


class JsonFileNotFound(Exception):
    pass


class JsonFileStructureIncorrect(Exception):
    pass