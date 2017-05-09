import pprint

if __name__ == '__main__':
    from core.net_interface import Net
    from core.net_corrector_nag import NAG
    from dataset.dataset import dataset
    from dataset.normalizer import normalize, normalize_vector

    net = Net(NAG)
    net.initialize_from('/Users/nikon/PycharmProjects/laperseptron/data/test.config.json')
    # net.load_from('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')

    test_dataset = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.test.csv')
    train_dataset = dataset('/Users/nikon/PycharmProjects/laperseptron/data/iris.train.csv')

    normalize(test_dataset)
    sigmas = normalize(train_dataset)
    net.set_deviation(sigmas)

    for vector in test_dataset:
        normalize_vector(vector[0], net)

    net.train(2000, train_dataset, test_dataset, 0.15)
    # net.trry(train_dataset, test_dataset)

    net.save_to('/Users/nikon/PycharmProjects/laperseptron/data/iris.net.json')
