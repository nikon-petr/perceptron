import pprint

if __name__ == '__main__':
    from core.net import Net
    from core.net_initializer import NetInitializer
    from core.net_loader import NetLoader
    from core.net_calculator import NetCalculator
    from core.net_error_evaluator import NetErrorEvaluator
    from core.net_corrector_cm import CM
    from core.net_corrector_nag import NAG

    from core.net_transfer_functions import sigmoid, d_sigmoid

    net = Net()
    net_loader = NetLoader()
    net_initializer = NetInitializer()

    net = net_loader.upload(net, 'test.config.json')
    net = net_initializer.initialize(net)
    net_loader.unload(net, 'test.json')

    input_vector = [-1, 1]
    expected_vector = [1, -1]

    net_calculator = NetCalculator(transfer_function=sigmoid, function_parameters={'alpha': 1})
    calculated_net = net_calculator.calculate(net, input_vector)

    net_error_evaluator = NetErrorEvaluator()
    error_evaluation = net_error_evaluator.evaluate(calculated_net, expected_vector)
    print('error evaluation =', error_evaluation)

    # cm_corrector = CM(nu=0.2, mu=0.975, d_transfer_function=d_sigmoid, function_parameters={'alpha': 1})
    nag_corrector = NAG(0.2, 0.975, sigmoid, {'alpha': 1}, d_sigmoid, {'alpha': 1})
    corrected_net = nag_corrector.correct_weights(calculated_net, expected_vector)

    pprint.pprint(corrected_net.layers, indent=4)

    calculated_again_net = net_calculator.calculate(corrected_net, input_vector)

    error_evaluation = net_error_evaluator.evaluate(calculated_again_net, expected_vector)
    print('error evaluation =', error_evaluation)

