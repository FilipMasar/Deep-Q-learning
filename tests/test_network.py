import numpy as np
import gym

from src.network import Network


def test_network_shape():
    env = gym.make("CartPole-v1")
    network = Network(env.observation_space.shape, 3, env.action_space.n, 0.01)

    assert (network._model.input_shape == (None, env.observation_space.shape[0]))
    assert (network._model.output_shape == (None, env.action_space.n))


def test_network_predict():
    env = gym.make("CartPole-v1")
    network = Network(env.observation_space.shape, 3, env.action_space.n, 0.01)

    states = [[1, 2, 3, 4]]
    q_values = network.predict(states)

    assert (isinstance(q_values, np.ndarray))
    assert (q_values.dtype == np.float32)
    assert (q_values.shape == (1, env.action_space.n))


def test_network_predict_on_batch():
    env = gym.make("LunarLander-v2")
    network = Network(env.observation_space.shape, 3, env.action_space.n, 0.01)
    network._model.summary()

    states = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
    q_values = network.predict(states)

    assert (isinstance(q_values, np.ndarray))
    assert (q_values.dtype == np.float32)
    assert (q_values.shape == (2, env.action_space.n))


def test_network_copy():
    env = gym.make("CartPole-v1")
    network = Network(env.observation_space.shape, 3, env.action_space.n, 0.01)
    network2 = Network(env.observation_space.shape, 3, env.action_space.n, 0.01)

    network.copy_weights_from(network2)

    for var, other_var in zip(network._model.variables, network2._model.variables):
        np.testing.assert_array_equal(var, other_var)
