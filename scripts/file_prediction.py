from models import load_model, predict
from data_processing import timestamp_to_features
from data_loader import load_data

import numpy as np

name = input()

times, _ = load_data('var_tec_reshape.npz')
model = load_model('models/%s' % name)
x = timestamp_to_features(times)
p = predict(x, model)

np.savez(name, times=times, var_tec_maps=p)
