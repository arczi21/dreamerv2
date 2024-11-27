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
    left = (x <= 128)
    down = (y >= 128)
    score0 = ram[:, :, 80].astype(int)
    score1 = ram[:, :, 82].astype(int)
    score2 = ram[:, :, 84].astype(int)
    score = (score0 / 5) * 100 + (score1 / 5) * 10 + score2 / 5
    score = (score >= 126.5)
    features = np.stack([left, down, score], axis=2)
    return features.astype(int)


def get_time(data):
    ram = data['ram'].numpy().astype(int)
    t = ram[:, :, 91] * 255 + ram[:, :, 90]
    return t

