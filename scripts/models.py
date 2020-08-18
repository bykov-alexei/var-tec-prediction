import numpy as np
from keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import load_model
import pickle as pkl


def load(filename):
    model = load_model('models/'+filename)
    return model


def save_model(model):
    model.save('models/'+model.name)
    return None


def dense_model(x, y, name='', hidden_layer_neurons=5, loss='mse', optimizer='adam'):
    inp = layers.Input(shape=x[0].shape)
    layer = layers.Dense(hidden_layer_neurons, activation='relu')(inp)
    layer = layers.Dense(np.prod(y[0].shape), activation='relu')(layer)
    layer = layers.Reshape((71, 73))(layer)

    model = models.Model(inp, layer, name=name)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def conv_model(x, y, name='', loss='mse', optimizer='adam'):
    inp = layers.Input(shape=x[0].shape)
    layer = layers.Dense(np.prod(y[0].shape), activation='relu')(inp)
    layer = layers.Reshape((y[0].shape[0], y[0].shape[1], 1))(layer)
    layer = layers.Conv2DTranspose(filters=5, kernel_size=5, padding='same')(layer)
    layer = layers.Conv2DTranspose(filters=1, kernel_size=5, padding='same')(layer)

    model = models.Model(inp, layer, name=name)
    model.compile(loss=loss, optimizer=optimizer)
    return model

def coef_model(x, y, name='', hidden_layer_neurons=5, loss='mse', optimizer='adam'):
    inp = layers.Input(shape=x[0].shape)
    layer = layers.Dense(hidden_layer_neurons)(inp)
    layer = layers.Dense(np.prod(y[0].shape), activation='relu')(layer)
    layer = layers.Reshape(y[0].shape)(layer)

    model = models.Model(inp, layer, name=name)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def coef_predict(model, x, eofs):
    pcs = model.predict(x)
    shape = list(eofs.shape)
    shape[0] = len(pcs)
    y = np.zeros(shape)
    for i in range(len(y)):
        m = np.zeros(eofs[0].shape)
        for j, eof in enumerate(eofs):
            m = m + pcs[i][j] * eof
        y[i] = m
    return y


def fit_model(x_train, x_test, y_train, y_test, model, epochs=1):
    reduce_rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    tensor_board = callbacks.TensorBoard(log_dir='./logs')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.2, patience=10, restore_best_weights=True)
    model_checkpoint = callbacks.ModelCheckpoint(filepath='models/weights/%s-epoch-{epoch}.h5' % model.name)

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        callbacks=[
                            reduce_rl,
                            tensor_board,
                            early_stopping,
                            model_checkpoint,
                        ])
    save_model(model)
    with open('models/histories/' + model.name + '.pkl', 'wb') as f:
        pkl.dump(history.history, f)
    return None


def predict(x, model):
    return model.predict(x)
