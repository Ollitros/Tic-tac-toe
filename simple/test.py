from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from tic_tac_toe.simple_tic_tac_toe import TicTacToe


def show_progress(batch_rewards_1, batch_rewards_2, log1, log2, percentile, reward_range=[-990, +100]):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward_1, threshold_1 = np.mean(batch_rewards_1), np.percentile(batch_rewards_1, percentile)
    log1.append([mean_reward_1, threshold_1])

    mean_reward_2, threshold_2 = np.mean(batch_rewards_2), np.percentile(batch_rewards_2, percentile)
    log2.append([mean_reward_2, threshold_2])

    print("mean reward = %.3f, threshold=%.3f" % (mean_reward_1, threshold_1))
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log1))[0], label='Mean rewards_1')
    plt.plot(list(zip(*log1))[1], label='Reward thresholds_1')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(batch_rewards_1, range=reward_range)
    plt.vlines([np.percentile(batch_rewards_1, percentile)], [0], [100], label="percentile_1", color='red')
    plt.legend()
    plt.grid()

    print("mean reward = %.3f, threshold=%.3f" % (mean_reward_2, threshold_2))
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log2))[0], label='Mean rewards_2')
    plt.plot(list(zip(*log2))[1], label='Reward thresholds_2')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(batch_rewards_1, range=reward_range)
    plt.vlines([np.percentile(batch_rewards_1, percentile)], [0], [100], label="percentile_2", color='red')
    plt.legend()
    plt.grid()

    plt.show()


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = [s for i in range(len(states_batch)) if rewards_batch[i] >= reward_threshold for s in
                    states_batch[i]]
    elite_actions = [a for i in range(len(actions_batch)) if rewards_batch[i] >= reward_threshold for a in
                     actions_batch[i]]

    return elite_states, elite_actions


def generate_session(t_max=10):
    states_1, actions_1 = [], []
    total_reward_1 = 0

    states_2, actions_2 = [], []
    total_reward_2 = 0

    s = env.reset()
    print("State-reset: ", s)

    for t in range(t_max):

        for agent in (agent_1, agent_2):

            # a vector of action probabilities in current state
            probs = agent.predict_proba([s])[0]
            a = np.random.choice(n_actions, 1, p=probs)[0]

            action = env.states
            if action[a] == 1 or action[a] == 2:
                continue

            step = 1
            if agent == agent_2:
                step = 2
            new_s, r, done = env.step(a, step)

            # record sessions like you did before
            if agent == agent_1:
                states_1.append(s)
                actions_1.append(a)
                total_reward_1 += r
            else:
                states_2.append(s)
                actions_2.append(a)
                total_reward_2 += r

            s = new_s

            print("new_state - ", new_s)
            print("r = ", r)

            if done:
                break
    return [states_1, actions_1, total_reward_1], [states_2, actions_2, total_reward_2]


env = TicTacToe(game_option=2)
env.reset()
n_actions = env.n
print("Actions: \n", env.actions)
print("Total number of actions: ", n_actions)

# Agent 1
agent_1 = MLPClassifier(hidden_layer_sizes=(20, 20),
                        activation='tanh',
                        warm_start=True,
                        max_iter=1 #make only 1 iteration on each .fit(...)
                        )

# initialize agent to the dimension of state an amount of actions
print([env.reset()]*n_actions)
agent_1.fit([env.reset()]*n_actions, range(n_actions))

# Agent 2
agent_2 = MLPClassifier(hidden_layer_sizes=(20, 20),
                        activation='tanh',
                        warm_start=True,
                        max_iter=1 #make only 1 iteration on each .fit(...)
                        )

# initialize agent to the dimension of state an amount of actions
print([env.reset()]*n_actions)
agent_2.fit([env.reset()]*n_actions, range(n_actions))

n_sessions = 100
percentile = 70
log1 = []
log2 = []

for i in range(50):
    print('\n\n\n !!! STEP - ', i+1)
    # generate new sessions
    results = [generate_session() for i in range(n_sessions)]

    # print(results)
    # print(results[0])
    # print(results[0][0])
    #
    # print(len(results))
    # print(len(results[0]))
    # print(len(results[0][0]))

    session_1 = list()
    session_2 = list()

    for i in range(n_sessions):
        # print(results[i][0])
        session_1.append(results[i][0])
        session_2.append(results[i][1])

    # Feed Agent 1
    batch_states, batch_actions, batch_rewards_1 = map(np.array, zip(*session_1))
    elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards_1, percentile)
    #     print(elite_states[:3])
    #     print(elite_actions[:3])
    agent_1.fit(elite_states, elite_actions)

    # Feed Agent 2
    batch_states, batch_actions, batch_rewards_2 = map(np.array, zip(*session_2))
    elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards_2, percentile)
    #     print(elite_states[:3])
    #     print(elite_actions[:3])
    agent_2.fit(elite_states, elite_actions)

    show_progress(batch_rewards_1, batch_rewards_2, log1, log2, percentile, reward_range=[0, np.max(batch_rewards_1)])
    # show_progress(batch_rewards_1, batch_rewards_2, log1, log2, percentile)
