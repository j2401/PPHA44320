import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sp
D = 3 * 10 ** 6
Q = 10 ** 5
N = 81
SD = 4
DELTA = 1 / 1.05
p_space = np.linspace(0, 80, N)
profit_c = p_space * Q - D
#cut-offs (N + 1)  
cut = np.linspace(0.5, 79.5, N-1)
#cut = np.insert(cut, 0, np.NINF)
cut = np.insert(cut, N-1, np.inf)
T = np.zeros((N,N))
for i in range(N):
    cd = sp.norm.cdf(cut, p_space[i], SD)
    probability = np.zeros(81)
    for j in range(1, N):
        probability[j] = cd[j] - cd[j-1]
    probability[0] = cd[0]
    T[i, :] = probability
V = np.zeros(N)
c = np.zeros(N)
Vnext = np.zeros((N,2))
count = 0
#probability * possible value
while True:
    for i in range(N):
        tommorrow = DELTA * np.dot(V, T[i, :])
        now = profit_c[i] * 1
        Vnext[i, 0] = np.max([now, tommorrow])
        Vnext[i, 1] = now >= tommorrow
    count += 1
    error = np.linalg.norm((Vnext[:, 0] - V), ord =2)
    if error <= 10 ** -8:
        break
    V = Vnext[:,0] * 1
a=Vnext[:,1]
b=np.min(np.where(a==1)[0])
print(p_space[b], count)
