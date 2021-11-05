import numpy as np
import sparse
from value_iter_2 import iter_value_2
from math import floor
from matplotlib import pyplot as plt
# Constants
S = 1000
RHO = 0.05
N = 501
DELTA = 1 / 1.05
action_space_2 = np.linspace(0, 1000 ** 0.5, N)
state_space = np.linspace(0, 1000, N)
action_space_1 = np.linspace(0, 1000, N)
# Coordinates of (non-zero) elements in the T(sparse) matrix, following the COO construction of sparse matrix.
xc = np.zeros(N*N).astype('int')
yc = np.zeros(N*N).astype('int')
zc = np.zeros(N*N).astype('int')
# Array of non-zero values in T. 
dc = np.zeros(N*N)
def u1(xc, yc):
    if yc > xc:
        return -10000
    return 2 * yc ** 0.5
def u2(xc, yc):
    if yc > xc:
        return -10000
    return 5 * yc - 0.05 * yc **2     
def construct_T(action, action_space, state_space, interpolation = True):
    x_co = np.full(N, action)
    y_co = np.arange(N)
    z_co = np.zeros(N)
    if interpolation == True:
        extraction = action_space[action] ** 2
        stock_n = state_space - extraction
        stock_n[stock_n < 0] = 0
        data = np.zeros(N)
        for i in range(N):
            state_n = stock_n[i]/2 # is either 0, positive integer, or positive float
            if floor(state_n) != state_n:
                b = floor(state_n)
                z_co[i] = b
                data[i] = 1 - (state_n - b) 
            else:
                data[i] = 1
    else:
        z_co = y_co - x_co
        z_co[z_co < 0] = 0
        data = np.full(N, 1) 
    return x_co, y_co, z_co, data
# Utility matrix
m1 = np.zeros((N,N))
m2 = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        m2[i, j] = u2(state_space[i], action_space_2[j] **2)
        m1[i, j] = u2(state_space[i] , action_space_1[j] )

for i in range(N):
    xc[i*N : (i+1)*N], yc[N*i : (i+1)*N], zc[N*i : (i+1)*N], dc[N*i : N*(i+1)] = construct_T(i, action_space_2, state_space, True)
action = iter_value_2(m2, xc, yc, zc, dc, N, DELTA).astype('int')
T_op = np.zeros((0,0))
x = np.arange(N)
y = x - action
y[y < 0] = 0
st = S
extraction = np.zeros(80)
interpolation = True
for i in range(80): 
    state = st / 2
    if interpolation:
        if floor(state) == state:
            idx = action[int(state)]
            extraction[i] = action_space_2[idx] **2
        else:
            b = floor(state)
            w1 = 1 - (state - b)
            w2 = 1 - w1 
            ex1 = action[b]
            ex2 = action[b+1] 
            extraction[i] = w1 * action_space_2[ex1] **2  + w2 * action_space_2[ex2] **2
    else:
        extraction[i] = action_space_1[action[int(state)]]
    st -= extraction[i]
plt.plot(np.arange(80), extraction)
plt.show()
