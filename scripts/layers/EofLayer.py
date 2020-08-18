import keras
import tensorflow as tf

class EofLayer(keras.layers.Layer):

    def __init__(self, eofs):
        super(EofLayer, self).__init__()
        self.eofs = tf.convert_to_tensor(eofs.astype('float32'))
        self.eofs.trainable = False
    
    def call(self, inputs):
        return tf.tensordot(inputs, self.eofs, axes=1)