import numpy as np
import tensorflow as tf
import keras
import keras.layers as L
import matplotlib.pyplot as plt
from tic_tac_toe.standart_tic_tac_toe import TicTacToe
from keras.models import save_model, load_model


class AgentApprox:
    def __init__(self):

        self.network = keras.models.Sequential()
        self.network.add(L.InputLayer(state_dim))

        # let's create a network for approximate q-learning following guidelines above

        self.network.add(L.Dense(100, activation='relu'))
        self.network.add(L.Dense(100, activation='relu'))
        self.network.add(L.Dense(n_actions))

        assert self.network.output_shape == (
        None, n_actions), "please make sure your model maps state s -> [Q(s,a0), ..., Q(s, a_last)]"
        assert self.network.layers[
                   -1].activation == keras.activations.linear, "please make sure you predict q-values without nonlinearity"

        # Create placeholders for the <s, a, r, s'> tuple and a special indicator for game end (is_done = True)
        self.states_ph = tf.placeholder('float32', shape=(None,) + state_dim)
        self.actions_ph = tf.placeholder('int32', shape=[None])
        self.rewards_ph = tf.placeholder('float32', shape=[None])
        self.next_states_ph = tf.placeholder('float32', shape=(None,) + state_dim)
        self.is_done_ph = tf.placeholder('bool', shape=[None])

        # get q-values for all actions in current states
        self.predicted_qvalues = self.network(self.states_ph)

        # select q-values for chosen actions
        self.predicted_qvalues_for_actions = tf.reduce_sum(self.predicted_qvalues * tf.one_hot(self.actions_ph, n_actions), axis=1)

        self.gamma = 0.99

        # compute q-values for all actions in next states
        self.predicted_next_qvalues = self.network(
             self.next_states_ph)  # <YOUR CODE - apply network to get q-values for next_states_ph>

        # compute V*(next_states) using predicted next q-values
        self.next_state_values = tf.reduce_max(self.predicted_next_qvalues, axis=1)

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        self.target_qvalues_for_actions = self.rewards_ph + self.gamma * self.next_state_values

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        self.target_qvalues_for_actions = tf.where(self.is_done_ph, self. rewards_ph, self.target_qvalues_for_actions)

        # mean squared error loss to minimize
        self.loss = (self.predicted_qvalues_for_actions - tf.stop_gradient(self.target_qvalues_for_actions)) ** 2
        self.loss = tf.reduce_mean(self.loss)

        # training function that resembles agent.update(state, action, reward, next_state) from tabular agent
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        assert tf.gradients(self.loss, [self.predicted_qvalues_for_actions])[
                   0] is not None, "make sure you update q-values for chosen actions and not just all actions"
        assert tf.gradients(self.loss, [self.predicted_next_qvalues])[
                   0] is None, "make sure you don't propagate gradient w.r.t. Q_(s',a')"
        assert self.predicted_next_qvalues.shape.ndims == 2, "make sure you predicted q-values for all actions in next state"
        assert self.next_state_values.shape.ndims == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert self.target_qvalues_for_actions.shape.ndims == 1, "there's something wrong with target q-values, they must be a vector"


def get_action(state, agent, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """

    q_values = agent.network.predict(state[None])[0]

    choice = np.random.random() > epsilon

    if choice:
        chosen_action = np.argmax(q_values)
    else:
        chosen_action = np.random.choice(n_actions)

    return chosen_action


def generate_session(agent_1, agent_2, step, t_max=30, epsilon=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward_1 = 0.0
    total_reward_2 = 0.0

    s = env.reset()

    done = False
    stop = False

    # Each agent has equal first steps in all games
    if step % 2 == 0:
        agents = (agent_1, agent_2)
        # print(step)
    else:
        agents = (agent_2, agent_1)
        # print(step)

    for t in range(t_max):
        for agent in agents:
            a = get_action(s, epsilon=epsilon, agent=agent)

            got = True

            while got:

                action = env.states

                nonzero = np.count_nonzero(action)
                if nonzero == n_actions:
                    stop = True
                    break
                if action[a] == 1 or action[a] == 2:
                    a = get_action(s, epsilon=epsilon, agent=agent)
                else:
                    got = False

            if stop:
                break

            step = 1
            if agent == agent_2:
                step = 2
            # print(agent, a)
            next_s, r, done = env.step(a, step)

            if train:
                sess.run(agent.train_step, {
                    agent.states_ph: [s], agent.actions_ph: [a], agent.rewards_ph: [r],
                    agent.next_states_ph: [next_s], agent.is_done_ph: [done]
                })

            s = next_s

            if agent == agent_1:
                total_reward_1 += r
            else:
                total_reward_2 += r

            if done:
                break

        if done or stop:
            break
    # print(total_reward_1, total_reward_2)
    return [total_reward_1], [total_reward_2]


tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

env = TicTacToe(game_mode=2)
env.reset()

n_actions = env.n
state_dim = np.shape(env.reset())
print(n_actions)
print(state_dim)


epsilon = 0.5

agent_1 = AgentApprox()
agent_2 = AgentApprox()

rewards_1 = []
rewards_2 = []
mean_rewards_1 = []
mean_rewards_2 = []
steps = []
for k in range(10):
    session_rewards = [generate_session(epsilon=epsilon, train=True, agent_1=agent_1,
                                        agent_2=agent_2, step=k) for _ in range(100)]

    rewards_1.append(session_rewards[0])
    rewards_2.append(session_rewards[1])

    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}\tAGENT".format(k, np.mean(rewards_1), epsilon), 1)
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}\tAGENT".format(k, np.mean(rewards_2), epsilon), 2)

    epsilon *= 0.99
    assert epsilon >= 1e-4, "Make sure epsilon is always nonzero during training"

    steps.append(k)
    mean_rewards_1.append(np.mean(rewards_1[-100:]))
    mean_rewards_2.append(np.mean(rewards_2[-100:]))
    plt.figure(figsize=[8, 4])
    plt.plot(steps, mean_rewards_1, label='Agent 1')
    plt.plot(steps, mean_rewards_2, label='Agent 2')
    plt.legend()
    plt.show()

# sessions = [generate_session(epsilon=epsilon, train=True, agent_1=agent_1,
#                              agent_2=agent_2, step=k) for _ in range(1000)]

# save_model(agent_1.network, 'data/models/approx_agent_1.h5')
# save_model(agent_2.network, 'data/models/approx_agent_2.h5')

