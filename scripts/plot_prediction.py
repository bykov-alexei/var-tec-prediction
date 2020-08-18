import pickle as pkl
import re
import os
from random import choice, choices

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import models

from data_loader import load_training_data
from models import predict, load_model

print(os.listdir('models'))
dense_raw = re.compile(r'dense_(\d+)_trained_on_raw_maps')
dense_reconstructed = re.compile(r'dense_(\d+)_trained_on_reconstructed_maps_neofs_(\d+)')
conv_raw = re.compile(r'conv_trained_on_raw_maps')
conv_reconstructed = re.compile(r'conv_trained_on_reconstructed_maps_neofs_(\d+)')

plots = 5

raw_data = np.load('data/raw_maps.npz')
raw_x, raw_y = raw_data['x'], raw_data['y']
indices = choices(list(range(len(raw_x))), k=plots)

for folder in os.listdir('models'):
    is_dense_raw = dense_raw.match(folder)
    is_dense_reconstructed = dense_reconstructed.match(folder)
    is_conv_raw = dense_raw.match(folder)
    print(folder)
    if is_dense_raw or is_conv_raw:
        fig, axs = plt.subplots(plots, 2)

        model = load_model('models/'+folder)
        predictions = predict(raw_x[indices], model)
        del model

        fig, axs = plt.subplots(plots, 2)
        fig.set_figwidth(16)
        fig.set_figheight(16)
        for i, ax in enumerate(axs):
            im = ax[0].imshow(raw_y[indices[i]])
            ax[0].set_title('raw_data')
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(predictions[i])
            ax[1].set_title('predicted')
            plt.colorbar(im, ax=ax[1])
        
        fig.savefig('predictions/%s.png' % folder)
        
    if is_dense_reconstructed:
        neofs = int(is_dense_reconstructed.group(2))

        data = np.load('data/reconstructed(neofs=%i).npz' % neofs)
        x, y = data['x'], data['y']
        model = load_model('models/'+folder)
        predictions = predict(x[indices], model)
        del model

        fig, axs = plt.subplots(plots, 3)
        fig.set_figwidth(16)
        fig.set_figheight(16)
        for i, ax in enumerate(axs):
            im = ax[0].imshow(raw_y[indices[i]])
            ax[0].set_title('raw_data')
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(y[indices[i]])
            ax[1].set_title('decomposed')
            plt.colorbar(im, ax=ax[1])

            ax[2].imshow(predictions[i])
            ax[2].set_title('predicted')
            plt.colorbar(im, ax=ax[2])

        del x
        del y
        fig.savefig('predictions/%s.png' % folder)

        keras.backend.clear_session()