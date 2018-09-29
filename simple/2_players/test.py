from IPython.display import clear_output
from utils.QLearningAgent import QLearningAgent
from utils.expected_value_sarsa_epsilon_annealing import EVSarsaAgent
from tic_tac_toe.simple_tic_tac_toe import TicTacToe
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
    !!!
    Q-learning in the wild + reducing epsilon
    !!!
"""


env = TicTacToe(game_mode=2)
env.reset()
n_actions = env.n
print(n_actions)
print(env)

agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                       get_legal_actions=lambda s: range(n_actions))


def play_and_train(env, agent, t_max=10 ** 4):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)  # <get agent to pick action given state s>

        action = env.states
        if action[a] == 1:
            continue

        next_s, r, done = env.step(a)

        # <train (update) agent for state s>
        agent.update(s, a, r, next_s)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


rewards = []
for i in range(6000):
    rewards.append(play_and_train(env, agent))
    if i % 1000 == 0:
        clear_output(True)
        print("mean reward",np.mean(rewards[-100:]))
        plt.plot(rewards)
        plt.show()

