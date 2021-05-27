# Reinforcement Learning Agent

Training agent using reinforcement learning algorithm to solve environments from open AI gym:

- CartPole-v1
- LunarLander-v2

Generally this agent can be applied to any environment with discrete set of actions and observation space described as a
1D list of numbers.
Algorithm implemented is off policy deep Q-learning with experience buffer

## Requirements

- Installing dependencies `pip3 install -r requirements.txt`
- It might cause some errors - check troubleshooting

## Running
Every time you `test` it you will get different results. That's because the environments use different seed for random number generator
To see the performance of trained agents run following command. By default, CartPole environment is chosen
```python
python3 main.py test
python3 main.py test --environment LunarLander-v2
```

Performing `train` action will override models in models/ dir. If you would like to see the performance of my trained models run `test` action first.
```python
python3 main.py train
python3 main.py train --environment LunarLander-v2

# you can also specify other arguments
python3 main.py train --environment CartPole-v1 --episodes 150 --batch_size 32 --gamma 1.0 --epsilon_start 0.5 --epsilon_final 0.1 --epsilon_final_at 500 --target_update_freq 0 --learning_rate 0.001 --hidden_layer_size 50
```

For executing tests, just run `pytest` in the root directory.

## Troubleshooting

When I was installing dependecies I run into few errors. Hopefully everything neccessary is in requirements.txt, but if
you encountered some problems feel free to reach out to me.

