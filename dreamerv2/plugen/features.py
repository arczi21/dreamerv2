import gym
import time
import numpy as np

env = gym.make('Breakout-v0')

"""
90 - time(???)
99 - x position
101 - y position
"""


def get_features(data):
    ram = data['ram'].numpy()
    x = ram[:, :, 99]
    y = ram[:, :, 101]
    left = (x <= 128)
    down = (y >= 128)
    features = np.stack([left, down], axis=2)
    return features

