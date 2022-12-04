import numpy as np
from numba import cuda

@cuda.jit
def vectorAddGPU(a, b, c):
    i = cuda.blockIdx.x
    c[i] = a[i] + b[i]

def main():
    N = 10

    A = np.arange(N, dtype=np.int32)
    B = np.arange(N, dtype=np.int32)
    C = cuda.device_array_like(A)

    nBlocks = 10
    nThreads = 1
    vectorAddGPU[nBlocks, nThreads](A,B,C)

    c=C.copy_to_host()
    print(c)

if __name__=='__main__':
    main()