import torch 
import torch.nn as nn
import numpy as np
from utils import drone_A, drone_B, drone_K, drone_smax, VFunc, UFunc
from tqdm import tqdm
import cvxpy as cp
from torch.autograd import grad, Variable


s_train = torch.from_numpy(np.random.uniform(low=-drone_smax, high=drone_smax, size=(100000, 8)).astype(np.float32)).cuda()
s_test = torch.from_numpy(np.random.uniform(low=-drone_smax, high=drone_smax, size=(10000, 8)).astype(np.float32)).cuda()

drone_A_th = torch.from_numpy(drone_A.astype(np.float32)).cuda()
drone_B_th = torch.from_numpy(drone_B.astype(np.float32)).cuda()
drone_K_th = torch.from_numpy(drone_K.astype(np.float32)).cuda()

v_func = VFunc(input_dim=8).cuda()
u_func = UFunc(input_dim=8, output_dim=3).cuda()

optim = torch.optim.Adam(list(v_func.parameters()) + list(u_func.parameters()), lr=1e-4, weight_decay=1e-3)

STEPS = 10000
dt = 0.02
beta = 0.5
alpha = 0.1
gamma = 2.0

batch_size = 1024

for i in range(STEPS):

    indices = np.random.randint(low=0, high=s_train.shape[0], size=batch_size)
    s_batch = s_train[indices]
    s_batch = Variable(s_batch, requires_grad=True).cuda()

    v = v_func(s_batch)
    u = u_func(s_batch)

    u_ref = torch.matmul(s_batch, -drone_K_th.permute(1, 0))
    s_dot = torch.matmul(s_batch, drone_A_th.permute(1, 0)) + torch.matmul(u, drone_B_th.permute(1, 0))

    # pvps = torch.cat([ 
    #     grad(v[j], s_batch, create_graph=True)[0][j].reshape(1, 8) for j in range(s_batch.shape[0])
    # ], dim=0)
    # pvps_sdot = torch.sum(pvps * s_dot, dim=1)

    v_dot = (v_func(s_batch + s_dot * dt) - v) / dt

    loss_vdot = torch.mean(nn.ReLU()(v_dot + alpha * v))
    loss_uref = torch.mean((u - u_ref) ** 2)
    loss_lbound = torch.mean(nn.ReLU()(torch.sum(s_batch ** 2, dim=1) * beta - v))
    loss_ubound = torch.mean(nn.ReLU()(v - torch.sum(s_batch ** 2, dim=1) * gamma))

    ref_w = 1.0 / (1 + np.exp(3.0 * (i * 1.0 / STEPS - 0.5)))
    loss = loss_vdot + loss_uref * ref_w + loss_lbound + loss_ubound

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    loss_np = loss.detach().cpu().numpy()
    print('Step: {} / {}, Loss: {:.4f}'.format(i, STEPS, loss_np))

torch.save(v_func.state_dict(), 'v_func.pth')
torch.save(u_func.state_dict(), 'u_func.pth')

exit(0)

print('==================================================')

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

