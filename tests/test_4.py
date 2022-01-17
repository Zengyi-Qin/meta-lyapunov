import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import multiprocessing as mp
import time

from numpy.ma.core import dot

IMG_PATH = 'canvas/canvas.png'
NUM_AGENTS = 20
dt = 0.01


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


def get_formation(q):
    img_prev = None
    while True:
        time.sleep(0.1)
        try:
            img = np.array(Image.open(IMG_PATH).convert('L'))
            #if img_prev is None or np.any(img_prev != img):
            dots = np.where(img < 128)
            dots = np.concatenate([dots[0][:, np.newaxis], dots[1][:, np.newaxis]], axis=1)
            img_prev = img
            q.put(dots)
        except:
            continue


def normalize(x):
    return x / (1e-9 + np.linalg.norm(x))


def res_distance(x1, x2, dist=1):
    origin = normalize(x1[:2] - x2[:2]) * dist + x2[:2]
    res = np.concatenate([x1[:2] - origin, x1[2:] - x2[2:]])
    return res


def render_init(num_agents):
    fig = plt.figure(figsize=(7, 5))
    return fig


q = mp.Queue()
p = mp.Process(target=get_formation, args=(q,))
p.start()

plt.ion()
plt.close()
fig = render_init(0)

dots_prev = None
states = np.zeros((NUM_AGENTS, 4))
dots_target = None

k = 0

while True:
    if not q.empty():
        dots = q.get()
    if len(dots) < NUM_AGENTS:
        continue

    if dots_prev is None:
        indices = np.linspace(0, len(dots), NUM_AGENTS, False).astype(np.int32)
        states[:, :2] = np.copy(dots[indices])
        dots_target = np.copy(dots[indices])
        dots_prev = dots

    if dots_prev.shape != dots.shape:
        indices = np.linspace(0, len(dots), NUM_AGENTS, False).astype(np.int32)
        dots_target = np.copy(dots[indices])
        dots_prev = dots 


    sdot_batch = np.zeros((NUM_AGENTS, 4), dtype=np.float32)

    for i in range(NUM_AGENTS):
        u_mean = 0
        for j in range(NUM_AGENTS):
            u1 = K.dot(res_distance(states[i], states[j], np.linalg.norm(dots_target[i] - dots_target[j])))
            u2 = K.dot(states[i] - np.concatenate([dots_target[i], np.zeros((2,))]))
            u_mean = (u1+u2)/2 + u_mean
        u_mean = u_mean / NUM_AGENTS
        sdot = A.dot(states[i]) + B.dot(u_mean)
        sdot_batch[i] = sdot 

    states = states + sdot_batch * dt

    k = k + 1
    if np.mod(k, 50) != 0:
        continue

    plt.clf()
    plt.scatter(states[:, 1], -states[:, 0], s=100, c='darkred')

    u_c, v_c = np.mean(states[:, :2], axis=0)
    x_c = v_c 
    y_c = -u_c

    plt.xlim(x_c-300, x_c+300)
    plt.ylim(y_c-214, y_c+214)

    fig.canvas.draw()
    plt.pause(0.01)
    
plt.clf()
