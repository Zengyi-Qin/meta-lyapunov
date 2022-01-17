import torch 
import torch.nn as nn
import numpy as np
from utils import drone_A, drone_B, drone_K, drone_smax, VFunc, UFunc
from tqdm import tqdm
import cvxpy as cp
from torch.autograd import grad, Variable


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


s = np.random.uniform(low=-drone_smax * 0.2, high=drone_smax * 0.2, size=(8, )).astype(np.float32)

for i in range(1000):
    s_input = Variable(torch.from_numpy(s[None].astype(np.float32)), requires_grad=True).cuda()
    v_output = v_func(s_input)
    v = v_output.detach().cpu().numpy().squeeze()
    pvps = grad(v_output, s_input)[0].detach().cpu().numpy().squeeze()

    L_A = pvps.dot(drone_A).dot(s)
    L_B = pvps.dot(drone_B)
    u_infer = u_func(torch.from_numpy(s[None].astype(np.float32)).cuda()).detach().cpu().numpy().squeeze()
    u_ref = -drone_K.dot(s)

    u = cp.Variable(3)
    slack = cp.Variable(1, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(u - u_infer) + slack)
    constraint = [ 
                L_A + L_B @ u + alpha * v <= slack
    ]
    prob = cp.Problem(objective, constraint)
    prob.solve()
    u = u.value

    print('Norm s: {:.4f}, V: {:.4f}'.format(np.linalg.norm(s), v))

    s_dot = drone_A.dot(s) + drone_B.dot(u)
    s = s + s_dot * dt

print('==================================================')