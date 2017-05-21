from math import e


class TransferFunctions:
    def __init__(self):
        self.__functions = {
            'sigmoid': {
                'f': lambda s, **kwargs: 1 / (1 + e ** (-(kwargs['alpha'] * s))),
                'df': lambda s, **kwargs: kwargs['alpha'] * s * (1 - s)
            },
            'tanh': {
                'f': lambda s, **kwargs: (2 / (1 + e ** (-2 * kwargs['alpha'] * s))) - 1,
                'df': lambda s, **kwargs: 1 - s ** 2
            }
        }

    def get_function(self, name):
        return self.__functions[name]['f'], self.__functions[name]['df']


def sigmoid(s, **kwargs):
    return 1 / (1 + e ** (-(kwargs['alpha'] * s)))


def d_sigmoid(s,  **kwargs):
    return kwargs['alpha'] * s * (1 - s)


def tanh(s, **kwargs):
    return (2 / (1 + e ** (-2 * kwargs['alpha'] * s))) - 1


def d_tanh(s, **kwargs):
    return 1 - s ** 2
