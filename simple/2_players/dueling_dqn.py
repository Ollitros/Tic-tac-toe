from tic_tac_toe.simple_tic_tac_toe import TicTacToe
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import deque


env = TicTacToe(game_mode=2)
env.reset()

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        cond = True
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            action = self.forward(state)

        else:
            cond = False
            action = list(range(env.n))
        return action, cond


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(batch_size, agent_1=False, agent_2= False ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    if agent_1:
        optimizer_1.zero_grad()
        loss.backward()
        optimizer_1.step()
    else:
        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()

    return loss


def plot(frame_idx, rewards_1, rewards_2, losses_1, losses_2):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Rewards')
    plt.plot(rewards_1, label=('Agent 1 | frame %s | reward: %s' % (frame_idx, np.mean(rewards_1[-10:]))))
    plt.plot(rewards_2, label=('Agent 2 | frame %s | reward: %s' % (frame_idx, np.mean(rewards_2[-10:]))))
    plt.legend()
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses_1, label='Agent 1')
    plt.plot(losses_2, label='Agent 2')
    plt.legend()
    plt.show()


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


current_model_1 = DuelingDQN(env.n, env.n)
target_model_1 = DuelingDQN(env.n, env.n)

current_model_2 = DuelingDQN(env.n, env.n)
target_model_2 = DuelingDQN(env.n, env.n)


if USE_CUDA:
    current_model_1 = current_model_1.cuda()
    target_model_1 = target_model_1.cuda()
    current_model_2 = current_model_2.cuda()
    target_model_2 = target_model_2.cuda()


optimizer_1 = optim.Adam(current_model_1.parameters())
optimizer_2 = optim.Adam(current_model_2.parameters())

replay_buffer = ReplayBuffer(1000)

update_target(current_model_1, target_model_1)
update_target(current_model_2, target_model_2)

# Training

num_frames = 10000
batch_size = 32
gamma = 0.99

losses_1 = []
losses_2 = []
all_rewards_1 = []
all_rewards_2 = []
episode_reward_1 = 0
episode_reward_2 = 0

state = env.reset()
for frame_idx in range(1, num_frames + 1):

    if frame_idx % 2 == 0:
        agents = ([current_model_1, target_model_1], [current_model_2, target_model_2])
        print(frame_idx)
    else:
        agents = ([current_model_2, target_model_2], [current_model_1, target_model_1])
        print(frame_idx)

    done = False
    for current_model, target_model in agents:

        epsilon = epsilon_by_frame(frame_idx)
        actions, cond = current_model.act(state, epsilon)
        a = env.states

        nonzero = np.count_nonzero(a)
        if nonzero == env.n:
            state = env.reset()
            break

        if cond:

            actions = actions.tolist()

            while True:
                maximum = np.max(actions)
                index_max = np.argmax(actions)

                if a[index_max] == 1 or a[index_max] == 2:
                    actions[0][index_max] = -10000
                else:
                    action = index_max
                    break

        else:
            while True:
                action = random.choice(actions)
                if a[action] == 1 or a[action] == 2:
                    continue
                else:
                    break

        print(action, cond)

        action = int(action)

        step = 1
        if current_model == current_model_2:
            step = 2

        next_state, reward, done = env.step(action, step)
        print(next_state)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state

        if current_model == current_model_1:
            episode_reward_1 += reward
        else:
            episode_reward_2 += reward

        if done:
            if current_model == current_model_1:
                all_rewards_1.append(episode_reward_1)
                episode_reward_1 = 0
            else:
                all_rewards_2.append(episode_reward_2)
                episode_reward_2 = 0

            state = env.reset()

        if current_model == current_model_1:
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size, agent_1=True)
                losses_1.append(loss.data[0])
        else:
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size, agent_2=True)
                losses_2.append(loss.data[0])

    if frame_idx % 200 == 0:
        plot(frame_idx, all_rewards_1, all_rewards_2, losses_1, losses_2)
    if frame_idx % 100 == 0:
        update_target(current_model_1, target_model_2)
        update_target(current_model_1, target_model_2)