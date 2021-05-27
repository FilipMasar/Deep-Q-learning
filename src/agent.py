import collections
import gym
import numpy as np
import matplotlib.pyplot as plt

from src.network import Network
from src.utils import calculate_training_stats


class Agent:
    def __init__(self, environment="CartPole-v1", episodes=100, batch_size=32, gamma=1.0, epsilon_start=0.5,
                 epsilon_final=0.1, epsilon_final_at=300, target_update_freq=0, learning_rate=0.001,
                 hidden_layer_size=50, model=None):

        self.env = gym.make(environment)
        self.env_eval = gym.make(environment)
        self.model_name = model if model else self.env.spec.id

        # construct networks
        self.network = Network(
            input_layer_size=self.env.observation_space.shape,
            hidden_layer_size=hidden_layer_size,
            output_layer_size=self.env.action_space.n,
            learning_rate=learning_rate
        )
        if target_update_freq:
            self.target_network = Network(
                input_layer_size=self.env.observation_space.shape,
                hidden_layer_size=hidden_layer_size,
                output_layer_size=self.env.action_space.n,
                learning_rate=learning_rate
            )
        else:
            self.target_network = self.network

        self.training_episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_final_at = epsilon_final_at

        self.training_progress = []

    def train(self):
        print("Training 💪 ...")
        replay_buffer = collections.deque(maxlen=100000)
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

        epsilon = self.epsilon_start
        for episode in range(self.training_episodes):
            # Perform episode
            state = self.env.reset()
            done = False
            returns = 0
            while not done:
                # Choose an action.
                q_values = self.network.predict([state])[0]
                action = np.argmax(q_values) if np.random.uniform() >= epsilon else self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                returns += reward

                # train a network on a batch of uniformly randomly chosen transitions.
                if len(replay_buffer) >= self.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=self.batch_size, replace=False)

                    states = np.zeros((self.batch_size, self.env.observation_space.shape[0]), np.float32)
                    next_states = np.zeros_like(states)
                    actions = np.zeros([self.batch_size], dtype=np.int32)
                    for i in range(self.batch_size):
                        states[i] = replay_buffer[batch[i]].state
                        next_states[i] = replay_buffer[batch[i]].next_state
                        actions[i] = replay_buffer[batch[i]].action

                    q_values = self.network.predict(states)
                    next_q_values = self.target_network.predict(next_states)
                    for i in range(self.batch_size):
                        transition = replay_buffer[batch[i]]
                        q_values[i][transition.action] = transition.reward + (
                            0 if transition.done else self.gamma * np.max(next_q_values[i]))

                    self.network.train(states, q_values)

                state = next_state

            # track progress
            self.training_progress.append(returns)

            # Copy to target network
            if self.target_update_freq and episode % self.target_update_freq == 0:
                self.target_network.copy_weights_from(self.network)

            # epsilon decay
            if self.epsilon_final_at:
                epsilon = np.interp(episode + 1, [0, self.epsilon_final_at], [self.epsilon_start, self.epsilon_final])

            # show stats
            if episode != 0 and episode % 50 == 0:
                returns = self.test(10)
                print(f'Evaluation after episode {episode} returned {returns}')

    def test(self, number_of_trials, render=False):
        returns = 0
        for _ in range(number_of_trials):
            state, done = self.env_eval.reset(), False
            while not done:
                if render:
                    self.env_eval.render()

                # choose the action that has the highest expected return
                action = np.argmax(self.network.predict([state])[0])
                state, reward, done, _ = self.env_eval.step(action)
                returns += reward / number_of_trials

        if render:
            self.env.close()

        return returns

    def plot_training_stats(self):
        x, mean, std = calculate_training_stats(self.training_progress, 50)

        plt.plot(x, mean)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.xlabel('Episode')
        plt.ylabel(f'Average returns over 50 episodes')
        plt.show()

    def save(self):
        self.network.save('models/' + self.model_name)

    def load(self):
        self.network.load('models/' + self.model_name)
