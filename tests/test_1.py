import torch
from torch import nn
import numpy as np


class Lyapunov(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 16)

        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        v = torch.mean(self.fc4(x) ** 2)
        return v 


class Controller(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        u = self.fc4(x)
        return u


v_func = Lyapunov(input_dim=4)
u_func = Controller(input_dim=4, output_dim=2)


optim_v = torch.optim.Adam(v_func.parameters(), lr=1e-4, weight_decay=1e-4)
optim_u = torch.optim.Adam(u_func.parameters(), lr=1e-4, weight_decay=1e-4)


x_start = np.concatenate([np.random.uniform(-1, 1, size=2), np.random.uniform(-0.2, 0.2, size=2)]).reshape(1, 4)

x = torch.from_numpy(x_start)