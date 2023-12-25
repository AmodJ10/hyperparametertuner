import numpy as np

class Individual:
    def __init__(self,params):
        self.params = params
        self.gene = list(params.values())
        self.score = -1
