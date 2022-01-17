import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([0, 0.5, 1, 0])
x2 = np.array([0, -0.5, 0, -1])
x3 = np.array([0.86, 0, -1, 0])

A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

B = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])

K = np.array([
    [-1, 0, -1, 0],
    [0, -1, 0, -1]
])

dt = 0.01

def normalize(x):
    return x / (1e-9 + np.linalg.norm(x))

def res_distance(x1, x2):
    origin = normalize(x1[:2] - x2[:2]) + x2[:2]
    res = np.concatenate([x1[:2] - origin, x1[2:] - x2[2:]])
    return res

def render_init(num_agents):
    fig = plt.figure(figsize=(7, 7))
    return fig


plt.ion()
plt.close()
fig = render_init(0)

for i in range(10000):
    u1_2 = K.dot(res_distance(x1, x2))
    u1_3 = K.dot(res_distance(x1, x3))
    u1 = (u1_2 + u1_3) / 2.0

    x1dot = A.dot(x1) + B.dot(u1)
    x1 = x1 + x1dot * dt

    u2_1 = K.dot(res_distance(x2, x1))
    u2_3 = K.dot(res_distance(x2, x3))
    u2 = (u2_1 + u2_3) / 2.0

    x2dot = A.dot(x2) + B.dot(u2)
    x2 = x2 + x2dot * dt

    u3_1 = K.dot(res_distance(x3, x1))
    u3_2 = K.dot(res_distance(x3, x2))
    u3 = (u3_1 + u3_2) / 2.0

    x3dot = A.dot(x3) + B.dot(u3)
    x3 = x3 + x3dot * dt


    if np.mod(i, 100) != 0:
        continue

    plt.clf()
    plt.scatter([x1[0]], [x1[1]], s=100, c='darkred')
    plt.scatter([x2[0]], [x2[1]], s=100, c='darkblue')
    plt.scatter([x3[0]], [x3[1]], s=100, c='darkgreen')

    x_c = (x1[0] + x2[0] + x3[0]) / 3.0
    y_c = (x1[1] + x2[1] + x3[1]) / 3.0

    plt.xlim(x_c-3, x_c+3)
    plt.ylim(y_c-3, y_c+3)

    fig.canvas.draw()
    plt.pause(0.01)
    
plt.clf()