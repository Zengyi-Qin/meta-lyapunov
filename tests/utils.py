import numpy as np
from numpy.linalg.linalg import norm
import torch 
import torch.nn as nn


drone_A = np.array([[0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

drone_B = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])

drone_K = np.array([[1, 0, 0, 2.41,    0,    0, 2.41,    0],
                    [0, 1, 0,    0, 2.41,    0,    0, 2.41],
                    [0, 0, 1,    0,    0, 1.73,    0,    0]])

drone_smax = np.array([10, 10, 10, 1, 1, 1, 0.2, 0.2])


def normalize(x, axis=0):
    return x / (1e-9 + np.linalg.norm(x, axis=axis, keepdims=True))


def residual(x, y, dist, n_pos=3):
    origin = normalize(x[:n_pos] - y[:n_pos]) * dist + y[:n_pos]
    res = np.concatenate([x[:n_pos] - origin, x[n_pos:] - y[n_pos:]])
    return res


def residual_batch(x, y_bch, dist_bch, n_pos=3):
    """
    Args:
        x (n_state)
        y_bch (n_batch, n_state)
        dist_bch (n_batch)
    Return:
        res (n_batch, n_state)
    """
    origin = normalize(x[:n_pos] - y_bch[:, :n_pos], axis=1) * dist_bch[:, np.newaxis] + y_bch[:, :n_pos]
    res = np.concatenate([x[:n_pos] - origin, x[n_pos:] - y_bch[:, n_pos:]], axis=1)
    return res


class VFunc(nn.Module):

    def __init__(self, input_dim):
        super(VFunc, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 64, bias=False)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        v = torch.mean(x**2, dim=1)
        return v


class UFunc(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(UFunc, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, output_dim, bias=False)
        self.activation = nn.ReLU()
        self.output_activate = nn.Tanh()

    def forward(self, input):
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        u = self.fc3(x)
        return u