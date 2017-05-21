import pprint

from lib.colors import Colors

if __name__ == '__main__':
    from core.net_interface import Net
    from core.net_nag_corrector import NAG
    from core.net_standart_corrector import SC
    from core.net_transfer_functions import d_tanh
    from dataset.dataset import dataset
    from dataset.normalizer import calculate_normalize, normalize_vector, normalize_dataset

    net = Net(SC, corrector_param={'df': d_tanh, 'df_p': {'alpha': 1}})
    net.initialize_from('/Users/nikon/PycharmProjects/laperseptron/data/iris.config.json', 0.001)
    # net.load_from('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')

    train = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.train.csv')
    test = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.test.csv')

    net.set_deviation(calculate_normalize(train))
    normalize_dataset(train, net)
    normalize_dataset(test, net)

    net.train(20, 0.1, train, test, 0.1)

    print('\n%sTest Iris Setosa%s' % (Colors.OKGREEN if net.calculate(test[0][0]) == 1 else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Versicolour%s' % (Colors.OKGREEN if net.calculate(test[0][1]) == 2 else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Virginica%s' % (Colors.OKGREEN if net.calculate(test[0][2]) == 3 else Colors.FAIL, Colors.ENDC))
    print('\n%sTest Iris Setosa%s' % (Colors.OKGREEN if net.calculate(test[0][3]) == 1 else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Versicolour%s' % (Colors.OKGREEN if net.calculate(test[0][4]) == 2 else Colors.FAIL, Colors.ENDC))
    print('%sTest Iris Virginica%s' % (Colors.OKGREEN if net.calculate(test[0][5]) == 3 else Colors.FAIL, Colors.ENDC))

    net.save_to('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')
