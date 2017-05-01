from itertools import starmap


class NetErrorEvaluator:
    def evaluate(self, net_object, expected_output_vector):

        if net_object.layers is None:
            raise NetIsNotInitialized()
        if not net_object.is_calculated:
            raise NetIsNotCalculated()

        real_output_vector = [neuron['output'] for neuron in net_object.layers[-1]]

        if len(real_output_vector) != len(expected_output_vector):
            raise IncorrectExpectedOutputVectorLength()

        n = len(real_output_vector)
        error_evaluation = sum(starmap(lambda r, e: (r - e) ** 2, zip(real_output_vector, expected_output_vector))) / n

        return error_evaluation


class NetIsNotInitialized(Exception):
    pass


class NetIsNotCalculated(Exception):
    pass


class IncorrectExpectedOutputVectorLength(Exception):
    pass