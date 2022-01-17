import numpy as np
np.set_printoptions(3)
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import linalg
from utils import drone_A, drone_B, drone_K, drone_smax, residual_batch
import pdb


NUM_AGENTS = 20
dt = 0.01
S = linalg.solve_continuous_are(drone_A, drone_B, np.eye(8), np.eye(3))
i_mat = np.eye((8), dtype=np.float32) * 0.1


def get_formation(n_agent):
    pos = np.random.uniform(low=-10, high=10, size=(n_agent, 3))
    return pos 


def get_init_states(n_agent):
    pos = np.random.uniform(low=-10, high=10, size=(n_agent, 3))
    zeros = np.zeros((n_agent, 5))
    states = np.concatenate([pos, zeros], axis=1)
    return states


for _ in range(1):
    s = get_init_states(NUM_AGENTS)
    pos_ref = s[:, :3] + np.random.uniform(low=-2, high=2, size=(NUM_AGENTS, 3))
    dist_ref = np.linalg.norm(
        np.reshape(pos_ref, (-1, 3, 1)) - np.reshape(pos_ref.T, (1, 3, -1)), axis=1)

    for step in range(100000):

        sdot = np.zeros((NUM_AGENTS, 8))
        for i in range(NUM_AGENTS):
            res = residual_batch(s[i], s, dist_ref[i])

            k = np.argmax(np.sum(np.dot(res, S) * res, axis=1))
            w = np.ones((NUM_AGENTS))
            w[k] = 10

            L_A = np.sum(res.dot(S).dot(i_mat + drone_A) * res, axis=1)
            L_B = res.dot(S).dot(drone_B)

            u = cp.Variable(3)
            eps = cp.Variable(NUM_AGENTS, nonneg=True)

            objective = cp.Minimize(cp.sum_squares(u) + eps @ w)
            constraint = [ 
                L_A + L_B @ u <= eps
            ]
            prob = cp.Problem(objective, constraint)
            prob.solve()
            u = u.value
            eps = eps.value

            sdot[i] = drone_A.dot(s[i]) + drone_B.dot(u)

        s = np.clip(s + sdot * dt, -drone_smax, drone_smax)

        dist = np.linalg.norm(
            np.reshape(s[:, :3], (-1, 3, 1)) - np.reshape(s[:, :3].T, (1, 3, -1)), axis=1)
        msg = 'step: {}, dist: {:.5f}'.format(step, np.mean(np.abs(dist - dist_ref)))
        print(msg)