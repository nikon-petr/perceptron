import random

from core.net_calculator import NetCalculator
from core.net_error_evaluator import NetErrorEvaluator
from core.net_initializer import NetInitializer
from core.net_loader import NetLoader
from core.net_state import NetState

from lib.colors import Colors


def raise_exceptions(f):
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except:
            raise
    return wrapper


class Net:
    def __init__(self, corrector, corrector_param=None, calculator_param=None):
        self.__state = NetState()
        self.__loader = NetLoader()
        self.__initializer = NetInitializer()
        self.__calculator = NetCalculator(calculator_param) if calculator_param else NetCalculator()
        self.__error_evaluator = NetErrorEvaluator()
        self.__corrector = corrector(corrector_param) if corrector_param else corrector()

    @property
    def corrector(self):
        return self.__corrector

    @corrector.setter
    def corrector(self, new_corrector, new_corrector_param):
        self.__corrector = new_corrector(**new_corrector_param)

    @raise_exceptions
    def load_from(self, path):
        self.__state = self.__loader.upload(self.__state, path)

    @raise_exceptions
    def save_to(self, path):
        self.__loader.unload(self.__state, path)

    @raise_exceptions
    def initialize_from(self, path):
        self.__state = self.__loader.upload(self.__state, path)
        self.__state = self.__initializer.initialize(self.__state)

    @raise_exceptions
    def set_deviation(self, deviation_list):
        self.__state.deviation = deviation_list

    @raise_exceptions
    def get_diviation(self):
        return self.__state.deviation

    @raise_exceptions
    def calculate(self, vector):
        self.__state = self.__calculator.calculate(self.__state, vector)
        return self.__state.get_output_vector()

    def trry(self, train_dataset, test_dataset):
        self.__state = self.__calculator.calculate(self.__state, train_dataset[0][0], training=True)
        e = self.__error_evaluator.evaluate(self.__state, train_dataset[0][1])
        self.__state = self.__corrector.correct(self.__state, train_dataset[0][1])
        print(e)
        self.__state = self.__calculator.calculate(self.__state, train_dataset[0][0], training=True)
        e = self.__error_evaluator.evaluate(self.__state, train_dataset[0][1])
        self.__state = self.__corrector.correct(self.__state, train_dataset[0][1])
        print(e)

    @raise_exceptions
    def train(self, epoch, train_dataset, test_dataset, error_value):
        for epoch in range(epoch):
            e_sum = 0
            random.shuffle(train_dataset)
            for n, (input_vector, output_vector) in enumerate(train_dataset, start=1):
                self.__state = self.__calculator.calculate(self.__state, input_vector, training=True)
                e = self.__error_evaluator.evaluate(self.__state, output_vector)
                e_sum += e
                em = e_sum / n
                self.__state = self.__corrector.correct(self.__state, output_vector)

                e_color = Colors.OKGREEN if e < error_value else Colors.FAIL
                em_color = Colors.OKGREEN if em < error_value else Colors.FAIL

            print('EPOCH:%s train: %sEm = %.3f%s' % (epoch, em_color, em, Colors.ENDC))

            e_sum = 0
            random.shuffle(test_dataset)
            for n, (input_vector, output_vector) in enumerate(test_dataset, start=1):
                self.__state = self.__calculator.calculate(self.__state, input_vector)
                e = self.__error_evaluator.evaluate(self.__state, output_vector)
                e_sum += e
                em = e_sum / n

                e_color = Colors.OKGREEN if e < error_value else Colors.FAIL
                em_color = Colors.OKGREEN if em < error_value else Colors.FAIL

            print('EPOCH:%s test %sEm = %.3f%s' % (epoch, em_color, em, Colors.ENDC))

            # if e < error_value:
            #     break
