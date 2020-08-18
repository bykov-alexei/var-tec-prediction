import pickle as pkl
import re
import os
from random import choice, choices

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import models

from data_loader import load_training_data, load_reconstructed_training_data
from models import predict, load_model

def plot_losses(ax, loss, val_loss=None, name=None):
    ax.plot(loss, label='loss')
    if val_loss:
        ax.plot(val_loss, label='val_loss')
    if name:
        ax.set_title(name)
    ax.label()


def plot_img(ax, img, name=None, colorbar=True):
    im = ax.imshow(img)
    if colorbar:
        plt.colorbar(im, ax=ax)
    if name:
        ax.set_title(name)


def get_predictions(model, x):
    model = load_model('models/'+model)
    predictions = predict(x, model)
    del model
    return predictions


def save_fig(fig, title):
    fig.set_figwidth(16)
    fig.set_figheight(16)
    fig.suptitle(model)
    fig.savefig('predictions/%s.png' % model)

dense_raw = re.compile(r'dense_(\d+)_trained_on_raw_maps')
dense_reconstructed = re.compile(r'dense_(\d+)_trained_on_reconstructed_maps_neofs_(\d+)')
conv_raw = re.compile(r'conv_trained_on_raw_maps')
conv_reconstructed = re.compile(r'conv_trained_on_reconstructed_maps_neofs_(\d+)')
coef_dense_real = re.compile(r'coef_dense_(\d+)_trained_on_real_neofs_(\d+)')
coef_dense_reconstructed = re.compile(r'coef_dense_(\d+)_trained_on_real_neofs_(\d+)')

plots = 5

raw_data = np.load('data/raw_maps.npz')
raw_x, raw_y = raw_data['x'], raw_data['y']
indices = choices(list(range(len(raw_x))), k=plots)

for model in os.listdir('models'):
    is_dense_raw = dense_raw.match(model)
    is_dense_reconstructed = dense_reconstructed.match(model)
    is_conv_raw = dense_raw.match(model)
    is_coef_dense_real = coef_dense_real.match(model)
    is_coef_dense_reconstructed = coef_dense_reconstructed.match(model)
    print(model)
    match = None
    if is_dense_raw:
        match = is_dense_raw
    elif is_dense_reconstructed:
        match = is_dense_reconstructed
    elif is_conv_raw:
        match = is_conv_raw
    elif is_coef_dense_real:
        match = is_coef_dense_real
    elif is_coef_dense_reconstructed:
        match = is_coef_dense_reconstructed


    if is_dense_raw or is_conv_raw:
        fig, ax = plt.subplots(plots, 2)
        predictions = get_predictions(model, raw_x[indices])
        for i, index in enumerate(indices):
            plot_img(ax[i][0], raw_y[index], name='raw_data')
            plot_img(ax[i][1], predictions[i], name='prediction')

        save_fig(fig, model)
    
    if is_dense_reconstructed or is_coef_dense_reconstructed or is_coef_dense_real: 
        fig, ax = plt.subplots(plots, 3)

        neofs = int(match.group(2))
        
        x, y, pcs, eofs = load_reconstructed_training_data('data/reconstructed_maps(neofs=%i).npz' % neofs)
        predictions = get_predictions(model, raw_x[indices])

        for i, index in enumerate(indices):
            plot_img(ax[i][0], raw_y[index], name='raw_data')
            plot_img(ax[i][1], y[index], name='reconstructed')
            plot_img(ax[i][2], predictions[i], name='predicted')

        save_fig(fig, model)


    keras.backend.clear_session()