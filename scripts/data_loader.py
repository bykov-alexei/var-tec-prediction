import numpy as np
from sklearn import model_selection


def load_data(filename):
    data = np.load(filename)
    x, y = data['times'], data['var_tec_maps']
    return x, y


def load_training_data(filename):
    data = np.load(filename)
    x, y = data['x'], data['y']
    return x, y


def load_reconstructed_training_data(filename):
    data = np.load(filename)
    x, y, pcs, eofs = data['x'], data['y'], data['pcs'], data['eofs']
    return x, y, pcs, eofs


def train_test_split(x, y, real):
    indices = np.arange(len(x))
    indices_train, indices_test, y_train, y_test = model_selection.train_test_split(indices, y, random_state=42)
    return x[indices_train], x[indices_test], y_train, y_test, real[indices_train], real[indices_test]
    