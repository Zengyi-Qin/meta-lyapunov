import torch 
import torch.nn as nn
import numpy as np
from utils import drone_A, drone_B, drone_K, drone_smax, VFunc, UFunc, residual_batch
from tqdm import tqdm
import cvxpy as cp
from torch.autograd import grad, Variable


def get_init_states(n_agent):
    pos = np.random.uniform(low=-5, high=5, size=(n_agent, 3))
    zeros = np.zeros((n_agent, 5))
    states = np.concatenate([pos, zeros], axis=1)
    return states

NUM_AGENTS = 20
dt = 0.02
beta = 0.5
alpha = 0.1
gamma = 2.0

drone_A_th = torch.from_numpy(drone_A.astype(np.float32)).cuda()
drone_B_th = torch.from_numpy(drone_B.astype(np.float32)).cuda()
drone_K_th = torch.from_numpy(drone_K.astype(np.float32)).cuda()

v_func = VFunc(input_dim=8).cuda()
v_func.load_state_dict(torch.load('v_func.pth'))

u_func = UFunc(input_dim=8, output_dim=3).cuda()
u_func.load_state_dict(torch.load('u_func.pth'))

s = get_init_states(NUM_AGENTS)
pos_ref = s[:, :3] + np.random.uniform(low=-2, high=2, size=(NUM_AGENTS, 3))
dist_ref = np.linalg.norm(
    np.reshape(pos_ref, (-1, 3, 1)) - np.reshape(pos_ref.T, (1, 3, -1)), axis=1)

for step in range(100000):

    sdot = np.zeros((NUM_AGENTS, 8))
    for i in range(NUM_AGENTS):
        res = residual_batch(s[i], s, dist_ref[i])

        s_input = Variable(torch.from_numpy(res.astype(np.float32)), requires_grad=True).cuda()
        v_output = v_func(s_input)
        v = v_output.detach().cpu().numpy().squeeze()

        pvps = torch.cat([ 
            grad(v_output[j], s_input, create_graph=True)[0][j].reshape(1, 8) for j in range(s_input.shape[0])
        ], dim=0).detach().cpu().numpy()
        u_infer = np.mean(u_func(s_input).detach().cpu().numpy(), axis=0)

        L_f = np.sum(pvps * np.dot(res, drone_A.T), axis=1)
        L_g = np.dot(pvps, drone_B)

        u = cp.Variable(3)
        eps = cp.Variable(NUM_AGENTS, nonneg=True)

        objective = cp.Minimize(cp.sum_squares(u - u_infer) + cp.sum(eps))
        constraint = [ 
                L_f + L_g @ u <= eps
        ]
        prob = cp.Problem(objective, constraint)
        prob.solve()
        u = u.value
        eps = eps.value

        sdot[i] = drone_A.dot(s[i]) + drone_B.dot(u)

    s = s + sdot * dt

    dist = np.linalg.norm(
            np.reshape(s[:, :3], (-1, 3, 1)) - np.reshape(s[:, :3].T, (1, 3, -1)), axis=1)
    msg = 'step: {}, dist: {:.5f}'.format(step, np.mean(np.abs(dist - dist_ref)))
    print(msg)