import numpy as np
import torch 
from torch import nn 
from torch.autograd import Variable
import cvxpy as cp

from utils import drone_A, drone_B, drone_K, drone_smax

Q = Variable(torch.randn(3, 8).type(torch.FloatTensor), requires_grad=True)
optim = torch.optim.SGD([Q], lr=1e-3, momentum=0.9, weight_decay=1e-2)

s_train = torch.from_numpy(np.random.uniform(low=-drone_smax, high=drone_smax, size=(100000, 8)).astype(np.float32))
s_test = torch.from_numpy(np.random.uniform(low=-drone_smax, high=drone_smax, size=(10000, 8)).astype(np.float32))

alpha = torch.from_numpy(np.eye((8), dtype=np.float32) * 0.1)

drone_A_th = torch.from_numpy(drone_A.astype(np.float32))
drone_B_th = torch.from_numpy(drone_B.astype(np.float32))
drone_K_th = torch.from_numpy(drone_K.astype(np.float32))

for i in range(3000):
    Z = torch.matmul(torch.matmul(Q.permute(1, 0), Q), alpha + drone_A_th - torch.matmul(drone_B_th, drone_K_th))
    prod = torch.sum(torch.matmul(s_train, Z) * s_train, dim=1)
    loss = torch.mean(nn.ReLU()(prod))

    optim.zero_grad()
    loss.backward()
    optim.step()

Z = torch.matmul(torch.matmul(Q.permute(1, 0), Q), alpha + drone_A_th - torch.matmul(drone_B_th, drone_K_th))
prod = torch.sum(torch.matmul(s_test, Z) * s_test, dim=1).detach().numpy()
print(np.mean(prod < 0))
print(np.mean(prod))
print(Q)

Q = Q.detach().numpy()

i_mat = np.eye((8), dtype=np.float32) * 0.1

s = np.random.uniform(low=-drone_smax, high=drone_smax, size=(8,))
for i in range(10000):
    s = np.clip(s, -drone_smax, drone_smax)
    L_A = s.dot(Q.T).dot(Q).dot(i_mat + drone_A).dot(s)
    L_B = s.dot(Q.T).dot(Q).dot(drone_B)
    u = cp.Variable(3)
    constraint = [
        L_A + L_B @ u <= 0
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraint)
    prob.solve()
    u = u.value
    
    sdot = drone_A.dot(s) + drone_B.dot(u)
    s = s + sdot * 0.01
    print(np.linalg.norm(s))