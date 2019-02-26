import gym
import os

env = gym.make('Humanoid-v2')
env.reset()
env.render()