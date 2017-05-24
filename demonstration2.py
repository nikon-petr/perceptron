import itertools
import random

import numpy as np

if __name__ == '__main__':
    from lib.colors import Colors
    from core.net_interface import Net
    from core.net_nag_corrector import NAG
    from core.net_sgd_corrector import SGD
    from core.net_adam_corrector import Adam
    from core.net_transfer_functions import functions
    from dataset.dataset import dataset
    from dataset.normalizer import calculate_normalize, normalize_dataset

    adam = Adam(nu=0.01)
    net = Net(functions['tanh'], adam)
    # net.initialize_from('/Users/nikon/PycharmProjects/laperseptron/data/compressor.config.json', 0.001)
    net.load_from('/Users/nikon/PycharmProjects/laperseptron/data/compressor.net.json')
    #
    # train = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.train.csv')
    # test = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.test.csv')

    train = np.array((list(itertools.product([-1, 1], repeat=5)), list(itertools.product([-1, 1], repeat=5))))
    # test = np.array(([[random.uniform(-1, 1) for j in range(10)] for i in range(len(train))], [[random.uniform(-1, 1) for j in range(10)] for i in range(len(train))]))
    test = np.copy(train)

    net.train(500, 0.1, train, test, 200)

    net.save_to('/Users/nikon/PycharmProjects/laperseptron/data/compressor.net.json')

    print(net.calculate([-1, -1, 1, 0.8, 1], get_vector=True))
