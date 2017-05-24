if __name__ == '__main__':
    from lib.colors import Colors
    from core.net_interface import Net
    from core.net_nag_corrector import NAG
    from core.net_standart_corrector import SC
    from core.net_adam_corrector import Adam
    from core.net_transfer_functions import functions
    from dataset.dataset import dataset
    from dataset.normalizer import calculate_normalize, normalize_dataset

    net = Net(functions['tanh'], Adam)
    # net = Net(functions['tanh'], NAG, corrector_param={'mu': 0.97})
    # net = Net(functions['tanh'], SC)

    net.initialize_from('/Users/nikon/PycharmProjects/laperseptron/data/iris.config.json', 0.001)
    # net.load_from('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')

    train = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.train.csv')
    test = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.test.csv')

    net.set_normalization(calculate_normalize(train))
    normalize_dataset(train, net)
    normalize_dataset(test, net)

    net.train(5, 0.1, train, test, 2)
    # net.train(10, 0.1, train, test, 2)
    # net.train(25, 0.1, train, test, 5)

    net.save_to('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')

    print('\n%sTest Iris Setosa%s' % (Colors.OKGREEN if net.calculate(test[0][0]) == 'Iris Setosa' else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Versicolour%s' % (Colors.OKGREEN if net.calculate(test[0][1]) == 'Iris Versicolour' else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Virginica%s' % (Colors.OKGREEN if net.calculate(test[0][2]) == 'Iris Virginica' else Colors.FAIL, Colors.ENDC))
    print('\n%sTest Iris Setosa%s' % (Colors.OKGREEN if net.calculate(test[0][3]) == 'Iris Setosa' else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Versicolour%s' % (Colors.OKGREEN if net.calculate(test[0][4]) == 'Iris Versicolour' else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Virginica%s' % (Colors.OKGREEN if net.calculate(test[0][5]) == 'Iris Virginica' else Colors.FAIL, Colors.ENDC))
