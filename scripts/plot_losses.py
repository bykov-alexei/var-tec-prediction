import re
import os
import pickle as pkl
import matplotlib.pyplot as plt

from models import load_model

dense_raw = re.compile(r'dense_(\d+)_trained_on_raw_maps')
dense_reconstructed = re.compile(r'dense_(\d+)_trained_on_reconstructed_maps_neofs_(\d+)')
conv_raw = re.compile(r'conv_trained_on_raw_maps')
conv_reconstructed = re.compile(r'conv_trained_on_reconstructed_maps_neofs_(\d+)')
coef_dense_real = re.compile(r'coef_dense_(\d+)_trained_on_real_neofs_(\d+)')
coef_dense_reconstructed = re.compile(r'coef_dense_(\d+)_trained_on_reconstructed_neofs_(\d+)')


def find_matching(file):
    is_dense_raw = dense_raw.match(file)
    is_dense_reconstructed = dense_reconstructed.match(file)
    is_conv_raw = conv_raw.match(file)
    is_conv_reconstructed = conv_reconstructed.match(file)
    is_coef_dense_raw = coef_dense_real.match(file)
    is_coef_dense_reconstructed = coef_dense_reconstructed.match(file)

    match = None
    if is_dense_raw:
        match = 'dense_raw'
    elif is_dense_reconstructed:
        match = 'dense_reconstructed'
    elif is_conv_raw:
        match = 'conv_raw'
    elif is_conv_reconstructed:
        match = 'conv_reconstructed'
    elif is_coef_dense_raw:
        match = 'coef_dense_raw'
    elif is_coef_dense_reconstructed:
        match = 'coef_dense_reconstructed'
    return match


losses = {
    'dense_raw': {'x': [], 'y': []},
    'dense_reconstructed': {'x': [], 'y': []},
    'conv_raw': {'x': [], 'y': []},
    'conv_reconstructed': {'x': [], 'y': []},
    'coef_dense_raw': {'x': [], 'y': []},
    'coef_dense_reconstructed': {'x': [], 'y': []},
}

def save_fig(fig, title):
    fig.set_figwidth(16)
    fig.set_figheight(16)
    fig.suptitle(model)
    fig.savefig('predictions/%s.png' % model)

files = os.listdir('models/histories')
for file in files:
    try:
        with open(os.path.join('models', 'histories', file), 'rb') as f:
            history = pkl.load(f)
    except EOFError:
        continue
    print(file[:-4])
    model = load_model(os.path.join('models', file[:-4]))
    params = model.count_params()
    print(params)
    match = find_matching(file)
    losses[match]['x'].append(params)
    losses[match]['y'].append(history['val_loss'][-1])


fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

ax[0][0].scatter(losses['dense_raw']['x'], losses['dense_raw']['y'])
ax[1][0].scatter(losses['dense_reconstructed']['x'], losses['dense_reconstructed']['y'])

ax[0][1].scatter(losses['conv_raw']['x'], losses['conv_raw']['y'])
ax[1][1].scatter(losses['conv_reconstructed']['x'], losses['conv_reconstructed']['y'])

ax[0][2].scatter(losses['coef_dense_raw']['x'], losses['coef_dense_raw']['y'])
ax[1][2].scatter(losses['coef_dense_reconstructed']['x'], losses['coef_dense_reconstructed']['y'])

save_fig(fig, 'params-losses.png')
