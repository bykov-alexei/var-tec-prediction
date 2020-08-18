import numpy as np


def load_data(filename):
    data = np.load(filename)
    x, y = data['times'], data['var_tec_maps']
    return x, y


def load_training_data(filename):
    data = np.load(filename)
    x, y = data['x'], data['y']
    return x, y
