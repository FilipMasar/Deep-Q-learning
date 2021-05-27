import tensorflow as tf

from src.utils import convert_to_numpy


class Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate):
        self._model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_layer_size),
            tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_layer_size, activation=None),
        ])
        self._model.compile(
            optimizer=tf.optimizers.Adam(learning_rate),
            loss=tf.losses.MeanSquaredError()
        )

    @convert_to_numpy
    @tf.function  # for performance reasons
    def train(self, states, q_values):
        self._model.optimizer.minimize(
            lambda: self._model.loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    @convert_to_numpy
    @tf.function  # for performance reasons
    def predict(self, states):
        return self._model(states)

    # copy weights from a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)

    def save(self, path):
        self._model.save(path)

    def load(self, path):
        self._model = tf.keras.models.load_model(path)
