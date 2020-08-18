from data_loader import load_training_data
from models import conv_model, dense_model, fit_model

from sklearn.model_selection import train_test_split
from keras import optimizers

x, y = load_training_data('data/raw_maps.npz')
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

denses = [5, 20, 50, 100]

for hidden_layer_neurons in denses:
    model = dense_model(x, y,
                        hidden_layer_neurons=hidden_layer_neurons,
                        name='dense_%i_trained_on_raw_maps' % hidden_layer_neurons, optimizer=optimizers.Adam(learning_rate=0.004))

    fit_model(x_train, x_test, y_train, y_test, model, epochs=1000)

model = conv_model(x, y, name='conv_trained_on_raw_maps', optimizer=optimizers.Adam(learning_rate=0.004))
fit_model(x_train, x_test, y_train, y_test, model, epochs=1000)


for neofs in range(1, 30):
    x, y = load_training_data('data/reconstructed(neofs=%i).npz' % neofs)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    for hidden_layer_neurons in denses:
        model = dense_model(x, y,
                            hidden_layer_neurons=hidden_layer_neurons,
                            name='dense_%i_trained_on_reconstructed_maps_neofs_%i' % (hidden_layer_neurons, neofs), optimizer=optimizers.Adam(learning_rate=0.004))

        fit_model(x_train, x_test, y_train, y_test, model, epochs=1000)

    model = conv_model(x, y, name='conv_trained_on_reconstructed_maps_neofs_%i' % neofs, optimizer=optimizers.Adam(learning_rate=0.004))
    fit_model(x_train, x_test, y_train, y_test, model, epochs=1000)

