from numba.pycc import CC
import numpy as np
cc = CC('value_iter_2')
@cc.export('iter_value_2', 'f8[:](f8[:,:], i4[:], i4[:], i4[:], f8[:], i4, f8)')
def iter_value_2(m, x, y, z, d, N, DELTA):
    L = len(x)
    V = np.zeros(N)
    def next(m, v,x, y, z, d, N, DELTA):
        vnext = np.zeros((N,2))
        for i in range(L):
            value = m[y[i], x[i]] + DELTA * (d[i] * v[z[i]] + (1-d[i]) * v[z[i] +1])
            if vnext[y[i], 0] < value:
                vnext[y[i], 0] = value 
                vnext[y[i], 1] = x[i]
        return vnext
    while True:
        vp = next(m,V, x, y, z, d, N, DELTA)  
        value = vp[:,0]
        error = np.max(np.absolute(value-V))
        if error < 10 ** (-8):
            return vp[:,1]
        else:
            V = value * 1
cc.compile()