from core.net_error_evaluator import NetIsNotCalculated


class NetState:
    def __init__(self):
        self.config = None
        self.layers = None
        self.deviation = None
        self.is_calculated = False
        self.is_corrected = False

    def get_output_vector(self):

        if not self.is_calculated:
            raise NetIsNotCalculated

        return [neuron['output'] for neuron in self.layers[-1]]

    def clear_for_output(self):
        for layer in self.layers:
            for neuron in layer:
                del neuron['sigma']
                del neuron['input']
                del neuron['output']
                del neuron['d_input']
                del neuron['d_output']
                del neuron['velocity']
