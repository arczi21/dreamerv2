import gym
import numpy as np

env = gym.make('Breakout-v0')

"""
90 - time(???)
99 - x position
101 - y position
"""


def get_features(data):
    ram = data['ram'].numpy().astype(int)
    x = ram[:, :, 99]
    y = ram[:, :, 101]
    z = ram[:, :, 77]
    left = (x <= 128)
    down = (y >= 128)
    count = (z >= 40)
    features = np.stack([left, down, count], axis=2)
    return features.astype(int)


def get_time(data):
    ram = data['ram'].numpy().astype(int)
    t = ram[:, :, 91] * 255 + ram[:, :, 90]
    return t

