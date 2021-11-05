import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sp
import os
location = os.path.dirname(__file__)
D = 3 * 10 ** 6
Q = 10 ** 5
N = 81
SD = 4
DELTA = 1 / 1.05
p_space = np.linspace(0, 80, N)
profit_c = p_space * Q - D
# cut-offs (N + 1)  
cut = np.linspace(0.5, 79.5, N-1)
# cut = np.insert(cut, 0, np.NINF)
cut = np.insert(cut, N-1, np.inf)
T = np.zeros((N,N))
dummy = 1
# Contruct T
for i in range(N):
    # distribution is centered at current price, evaluated at cut
    cd = sp.norm.cdf(cut, p_space[i], SD)
    probability = np.zeros(81)
    for j in range(1, N):
        probability[j] = cd[j] - cd[j-1]
    probability[0] = cd[0]
    T[i, :] = probability
V = np.zeros(N)
Vnext = np.zeros((N,2))
# probability * possible value
# Value iteration
while True:
    for i in range(N):
        tommorrow = DELTA * np.dot(V, T[i, :])
        # prevent array referrence, this should make a copy.
        now = profit_c[i] * dummy
        Vnext[i, 0] = np.max([now, tommorrow])
        Vnext[i, 1] = now >= tommorrow
    error = np.linalg.norm((Vnext[:, 0] - V), ord =2)
    if error <= 10 ** -8:
        break
    V = Vnext[:,0] * dummy
control = Vnext[:,1]
p = np.min(np.where(control==1))
print("Tigger price: ", p)
# Plot
plt.plot(Vnext[:p+1,0], label = 'Option value', color = 'b')
plt.plot(np.arange(41, 81), np.zeros(40), color = 'orange')
plt.plot([41],[Vnext[41, 0]], '-ob', markersize = 5, markerfacecolor = 'none')
plt.xlabel("P")
plt.ylabel("Value")
plt.title("Option value")
plt.savefig(location + '/q3_1.png')
plt.show()

#Another plot which I believe, makes sense here
plt.clf()
plt.plot(Vnext[:p+1,0], label = 'Option value', color = 'b')
plt.vlines(41, 0, 3 * 10**6, color = 'orange')
plt.xlabel("P")
plt.ylabel("Value")
plt.title("Option value")
plt.savefig(location + '/q3_2.png')
plt.show()
print("Tigger price: ", p)
