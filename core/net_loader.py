import os
from json import JSONDecodeError
from json import dump
from json import load

import numpy as np

from core.net_errors import JsonFileStructureIncorrect, JsonFileNotFound


def upload(net_object, path):
    if not os.path.isfile(path):
        raise JsonFileNotFound()

    try:
        with open(path, 'r') as file:
            deserialized_file = load(file)
            net_object.config = deserialized_file['config']
            net_object.tags = deserialized_file['tags']
            net_object.net = deserialized_file.get('net')
            net_object.deviation = deserialized_file.get('normalization')

            if net_object.net:
                for l in range(1, len(net_object.config)):
                    net_object.net[l - 1]['w'] = np.array(net_object.net[l - 1]['w'])
                    net_object.net[l - 1]['o'] = np.zeros((net_object.config[l]))

    except KeyError:
        raise JsonFileStructureIncorrect()
    except JSONDecodeError:
        raise


def unload(net_object, path):
    try:
        net_copy = []
        for l in range(len(net_object.net)):
            net_copy.append({'w': net_object.net[l]['w'].tolist()})

        with open(path, 'w') as file:
            file_dictionary = {
                'config': net_object.config,
                'tags': net_object.tags,
                'net': net_copy,
                'normalization': net_object.normalization
            }
            dump(file_dictionary, file, sort_keys=True, indent=4)
    except JSONDecodeError:
        raise
