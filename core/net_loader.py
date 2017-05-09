import os

from json import load
from json import dump
from json import JSONDecodeError
from copy import deepcopy


class NetLoader:
    def __init__(self):
        self._base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

    def upload(self, net_object, path, relative=False):
        if relative:
            fullpath = os.path.join(self._base_dir, path)
        else:
            fullpath = path

        if not os.path.isfile(fullpath):
            raise JsonFileNotFound()

        try:
            with open(fullpath, 'r') as file:
                deserialized_file = load(file)
                loaded_net = deepcopy(net_object)
                loaded_net.config = deserialized_file['config']
                loaded_net.layers = deserialized_file.get('layers')
                return loaded_net

        except KeyError:
            raise JsonFileStructureIncorrect()
        except JSONDecodeError:
            raise

    def unload(self, net_object, path, relative=False):
        if relative:
            fullpath = os.path.join(self._base_dir, path)
        else:
            fullpath = path

        try:
            with open(fullpath, 'w') as file:
                file_dictionary = {'config': net_object.config, 'layers': net_object.layers}
                dump(file_dictionary, file, sort_keys=True, indent=4)
        except JSONDecodeError:
            raise


class JsonFileNotFound(Exception):
    pass


class JsonFileStructureIncorrect(Exception):
    pass