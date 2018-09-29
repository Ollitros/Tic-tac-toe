from IPython.display import clear_output
from utils.QLearningAgent import QLearningAgent
from utils.expected_value_sarsa_epsilon_annealing import EVSarsaAgent
from tic_tac_toe.standart_tic_tac_toe import TicTacToe
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

agent_1 = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                         get_legal_actions=lambda s: range(n_actions))

agent_2 = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                         get_legal_actions=lambda s: range(n_actions))


def play_and_train(env, agent_1, agent_2, step, t_max=20):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward_1 = 0.0
    total_reward_2 = 0.0
    s = env.reset()

    done = False
    stop = False

    # Each agent has equal first steps in all games
    if step % 2 == 0:
        agents = (agent_1, agent_2)
        print(step)
    else:
        agents = (agent_2, agent_1)
        print(step)

    for t in range(t_max):
        for agent in agents:

            a = agent.get_action(s)  # <get agent to pick action given state s>
            got = True

            while got:

                action = env.states

                nonzero = np.count_nonzero(action)
                if nonzero == n_actions:
                    stop = True
                    break
                if action[a] == 1 or action[a] == 2:
                    a = agent.get_action(s)
                else:
                    got = False

            if stop:
                break

            # print(agent, a)
            step = 1
            if agent == agent_2:
                step = 2

            next_s, r, done = env.step(a, step)

            # <train (update) agent for state s>
            agent.update(s, a, r, next_s)

            s = next_s

            if agent == agent_1:
                total_reward_1 += r
            else:
                total_reward_2 += r

            if done:
                break

        if done or stop:
            break

    return [total_reward_1], [total_reward_2]


rewards_1 = []
rewards_2 = []
mean_rewards_1 = []
mean_rewards_2 = []
steps = []
for k in range(100001):
    results = play_and_train(env, agent_1, agent_2, step=k)
    print(results)
    rewards_1.append(results[0])
    rewards_2.append(results[1])
    if k % 500 == 0:
        clear_output(True)
        steps.append(k)
        mean_rewards_1.append(np.mean(rewards_1[-100:]))
        mean_rewards_2.append(np.mean(rewards_2[-100:]))
        print("mean reward 1", mean_rewards_1[-1])
        print("mean reward 2", mean_rewards_2[-1])
        plt.figure(figsize=[8, 4])
        plt.plot(steps, mean_rewards_1, label='Agent 1')
        plt.plot(steps, mean_rewards_2, label='Agent 2')
        plt.legend()
        plt.show()

