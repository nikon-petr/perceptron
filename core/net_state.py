from core.net_errors import NetIsNotCalculated


class NetState:
    def __init__(self, f, df, f_param):
        self.tags = None
        self.config = None
        self.net = None
        self.normalization = None
        self.f = f
        self.df = df
        self.f_param = f_param
        self.is_calculated = False

    def get_output_vector(self):

        if not self.is_calculated:
            raise NetIsNotCalculated

        return self.net[-1]['o']

    def get_tag(self):
        return self.tags[self.get_output_vector().argmax(axis=0)]

    def clear(self):
        for l in self.net:
            del l['v']
            del l['o']
            del l['s']
