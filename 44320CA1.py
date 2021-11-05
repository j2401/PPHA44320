import numpy as np
from value_iter import iter_value
from math import floor
from matplotlib import pyplot as plt
import os
import time
# Constants
S = 1000
N = 501
T = 80
DELTA = 1 / 1.05
STEP = S / (N-1)
location = os.path.dirname(__file__)
state_space = np.linspace(0, 1000, N)
action_space_2 = np.linspace(0, 1000 ** 0.5, N)
action_space_1 = np.linspace(0, 1000, N)
# Coordinates of (non-zero) elements in the T(sparse) matrix, following the COO construction of sparse matrix.
xc = np.zeros(N*N).astype('int')
yc = np.zeros(N*N).astype('int')
zc = np.zeros(N*N).astype('int')
# Array of non-zero values in T(transition probabilities). 
p = np.zeros(N*N)
# Utility functions

def u1(xc, yc):
    if yc > xc:
        return np.NINF
    return 2 * yc ** 0.5
def u2(xc, yc):
    if yc > xc:
        return np.NINF
    return 5 * yc - 0.05 * yc **2
# Prices
def p1(y):
    if y > 0:
        return y ** -0.5
    return None
def p2(y):
    return 5 - 0.1 * y
# Returns the smaller state and its weight(kind of hard-coding here as we know there are only two states involved)
def weight(x):
    b = floor(x)
    w1 = 1 - (x - b)
    return b, w1
# The transition "matrix", although I didn't really implement a matrix"
def construct_T(action, action_space, state_space, interpolation):
    x_co = np.full(N, action)
    y_co = np.arange(N)
    z_co = np.zeros(N)
    if interpolation:
        extraction = action_space[action] ** 2
        stock_n = state_space - extraction
        stock_n[stock_n < 0] = 0
        p = np.zeros(N)
        for i in range(N):
            state_n = stock_n[i] / STEP # is either 0, positive integer, or positive float
            if floor(state_n) != state_n:
                b, w1 = weight(state_n)
                z_co[i] = b
                p[i] = w1
            else:
                z_co[i] = int(state_n)
                p[i] = 1
    else:
        z_co = y_co - x_co
        z_co[z_co < 0] = 0
        p = np.full(N, 1) 
    return x_co, y_co, z_co, p
def utility_matrix(state, action, u, N, interpolation):
    if interpolation:
        action = action ** 2
    return u(state[:, None], action[None, :])

def simulate(s0, action, action_space,
             interpolation = False , t=80):
    st = s0
    extraction = np.zeros(t)
    for i in range(t): 
        state = st / STEP
        if interpolation:
            if floor(state) == state:
                idx = action[int(state)]
                extraction[i] = action_space[idx] **2
            else:
                b, w1 = weight(state)
                w2 = 1 - w1
                extraction[i] = w1 * action_space[action[b]] **2  + w2 * action_space[action[b+1]] **2
        else:
            extraction[i] = action_space[action[int(state)]]
        st -= extraction[i]
    return extraction
def plotit(path_1, path_2, price_1, price_2, t, figure_name, interpolation):
    plt.clf()
    plt.plot(np.arange(t), path_1, label = 'Extraction(u1)')
    plt.plot(np.arange(t), path_2, label = 'Extraction(u2)')
    plt.xlabel("t")
    plt.ylabel("Extraction")
    plt.title("Extraction path, interpolation = "+str(interpolation))
    plt.legend()
    plt.savefig(location + '/' + figure_name)
    plt.clf()
    plt.plot(np.arange(t), price_1, label = 'Price(u1)')
    plt.plot(np.arange(t), price_2, label = 'Price(u2)')
    plt.xlabel("t")
    plt.ylabel("P")
    plt.title("Price path, interpolation = " +str(interpolation))
    plt.legend()
    plt.savefig(location + '/price_' + figure_name)
def run(state_space, action_space, x, y,
        z, p, figure_name, 
        interpolation = False, t=80):
    # utility matrices
    m1 = utility_matrix(state_space, action_space, util_1, N, interpolation)
    m2 = utility_matrix(state_space, action_space, util_2, N, interpolation)
    # T
    for i in range(N):
        x[i*N : (i+1)*N], y[N*i : (i+1)*N], z[N*i : (i+1)*N], p[N*i : N*(i+1)] = construct_T(i, action_space, state_space, interpolation)
    # Optimal choice at each state
    action_1 = iter_value(m1, x, y, z, p, N, DELTA).astype('int')
    action_2 = iter_value(m2, x, y, z, p, N, DELTA).astype('int')
    path_1 = simulate(S, action_1, action_space, interpolation)
    print('u1 extraction in last period is: ', path_1[-1])
    path_2 = simulate(S, action_2, action_space, interpolation)
    print('u2 extraction in last period is: ', path_2[-1])
    price_1 = p_u1(path_1)
    price_2 = p_u2(path_2)
    plotit(path_1, path_2, price_1, price_2,
           t, figure_name, interpolation)

t0 = time.time() 
util_1 = np.vectorize(u1)
util_2 = np.vectorize(u2)
p_u1 = np.vectorize(p1)
p_u2 = np.vectorize(p2)
run(state_space, action_space_1, xc, yc, zc, p, 'q1.png')
run(state_space, action_space_2, xc, yc, zc, p, 'q2.png', interpolation=True)
print(time.time()-t0)
    
