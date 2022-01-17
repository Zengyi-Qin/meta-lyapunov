import numpy as np
import torch 
from torch import nn 
from torch.autograd import Variable
import cvxpy as cp
from scipy import linalg

from utils import drone_A, drone_B, drone_K, drone_smax

S = linalg.solve_continuous_are(drone_A, drone_B, np.eye(8), np.eye(3))

s = np.random.uniform(low=-drone_smax, high=drone_smax, size=(8,))
i_mat = np.eye((8), dtype=np.float32) * 0.1

for i in range(10000):
    s = np.clip(s, -drone_smax, drone_smax)
    L_A = s.dot(S).dot(i_mat + drone_A).dot(s)
    L_B = s.dot(S).dot(drone_B)
    u = cp.Variable(3)
    constraint = [
        L_A + L_B @ u <= 0
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(u)), constraint)
    prob.solve()
    u = u.value

    #print(u, s)
    
    sdot = drone_A.dot(s) + drone_B.dot(u)
    s = s + sdot * 0.01
    print(np.linalg.norm(s))