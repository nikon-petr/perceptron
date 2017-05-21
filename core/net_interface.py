import random
import numpy as np

from core.net_calculator import calculate
from core.net_nag_corrector import NAG
from core.net_error_evaluator import evaluate
from core.net_initializer import initialize
from core.net_loader import upload, unload
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
        self.__corrector = corrector(**corrector_param) if corrector_param else corrector()

    @property
    def corrector(self):
        return self.__corrector

    @corrector.setter
    def corrector(self, new_corrector, new_corrector_param):
        self.__corrector = new_corrector(**new_corrector_param)

    @raise_exceptions
    def load_from(self, path):
        upload(self.__state, path)

    @raise_exceptions
    def save_to(self, path):
        unload(self.__state, path)

    @raise_exceptions
    def initialize_from(self, path, factor):
        upload(self.__state, path)
        initialize(self.__state, factor)

    @raise_exceptions
    def set_deviation(self, deviation_list):
        self.__state.deviation = deviation_list

    @raise_exceptions
    def get_deviation(self):
        return self.__state.deviation

    @raise_exceptions
    def calculate(self, vector):
        calculate(self.__state, vector)
        return self.__state.get_output_vector().argmax(axis=0) + 1

    def trry(self, inp, outp):
        self.__corrector.nu = 1
        calculate(self.__state, inp, training=True)
        print(evaluate(self.__state, outp))
        self.__corrector.correct(self.__state, outp)
        calculate(self.__state, inp, training=True)
        print(evaluate(self.__state, outp))

    @raise_exceptions
    def train(self, epoch, start_nu, train_data, test_data, step):
        self.__corrector.nu = start_nu
        em = 0
        for epoch in range(epoch):
            if epoch + 1 % step == 0:
                self.__corrector.nu /= 10

            e_sum = 0
            train_data_indexes = random.sample(range(len(train_data[0])), len(train_data[0]))
            for d in train_data_indexes:
                calculate(self.__state, train_data[0][d], training=True)
                self.__corrector.correct(self.__state, train_data[1][d])

            for d in range(len(test_data)):
                calculate(self.__state, test_data[0][d])
                e_sum += evaluate(self.__state, test_data[1][d])

            delta = em - e_sum / len(test_data)
            em = e_sum / len(test_data)
            em_color = Colors.OKGREEN if em < 0.15 else Colors.FAIL

            print('EPOCH:%s %sEm = %.3f%s D = %s' % (epoch, em_color, em, Colors.ENDC, delta))

            # if delta < 0:
            #     self.__corrector.nu = nu / 10
            # if abs(delta) < error_delta:
            #     break
