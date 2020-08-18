from data_loader import load_training_data, load_reconstructed_training_data, train_test_split
from models import conv_model, dense_model, fit_model, coef_model

from keras import optimizers

raw_x, raw_y = load_training_data('data/raw_maps.npz')
x_train, x_test, y_train, y_test, _, _ = train_test_split(raw_x, raw_y, raw_y)

denses = [5, 20, 50, 100]
neofs = range(1, 30)
epochs = 1000

for hidden_layer_neurons in denses:
    model = dense_model(raw_x, raw_y,
                        hidden_layer_neurons=hidden_layer_neurons,
                        name='dense_%i_trained_on_raw_maps' % hidden_layer_neurons, optimizer=optimizers.Adam(learning_rate=0.004))

    fit_model(x_train, x_test, y_train, y_test, model, epochs=epochs)
 
model = conv_model(raw_x, raw_y, name='conv_trained_on_raw_maps', optimizer=optimizers.Adam(learning_rate=0.004))
fit_model(x_train, x_test, y_train, y_test, model, epochs=epochs)


for n in neofs:
    x, y, pcs, eofs = load_reconstructed_training_data('data/reconstructed_maps(neofs=%i).npz' % n)
    x_train, x_test, y_train, y_test, real_train, real_test = train_test_split(x, y, raw_y)

    for hidden_layer_neurons in denses:
        model = dense_model(x, y,
                            hidden_layer_neurons=hidden_layer_neurons,
                            name='dense_%i_trained_on_reconstructed_maps_neofs_%i' % (hidden_layer_neurons, n), optimizer=optimizers.Adam(learning_rate=0.004))

        fit_model(x_train, x_test, y_train, real_test, model, epochs=epochs)

        model = coef_model(x, y, eofs, 
                            hidden_layer_neurons=hidden_layer_neurons,
                            name='coef_dense_%i_trained_on_real_neofs_%i' % (hidden_layer_neurons, n), optimizer=optimizers.Adam(learning_rate=0.004))

        fit_model(x_train, x_test, real_train, real_test, model, epochs=epochs)

        model = coef_model(x, y, eofs,
                            hidden_layer_neurons=hidden_layer_neurons,
                            name='coef_dense_%i_trained_on_reconstructed_neofs_%i' % (hidden_layer_neurons, n), optimizer=optimizers.Adam(learning_rate=0.004))

        fit_model(x_train, x_test, y_train, real_test, model, epochs=epochs)

    model = conv_model(x, y, name='conv_trained_on_reconstructed_maps_neofs_%i' % n, optimizer=optimizers.Adam(learning_rate=0.004))
    fit_model(x_train, x_test, y_train, real_test, model, epochs=epochs)

