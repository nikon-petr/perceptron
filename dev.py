if __name__ == '__main__':
    from core.net import Net
    from core.net_initializer import NetInitializer
    from core.net_loader import NetLoader
    from core.net_calculator import NetCalculator

    from core.net_transfer_functions import sigmoid

    net = Net()
    net_loader = NetLoader()
    net_initializer = NetInitializer()

    net = net_loader.upload(net, 'test.config.json')
    net = net_initializer.initialize(net)
    net_loader.unload(net, 'test.json')

    net_calculator = NetCalculator(sigmoid)
    result = net_calculator.calculate(net, [-1, 1])
