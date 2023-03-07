import numpy as np
from numba import cuda

@cuda.jit
def vectorAddGPU(a, b, c):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    c[i] = a[i] + b[i]

def main():
    N = 10000

    A = np.ones(N, dtype=np.int32)
    B = np.ones(N, dtype=np.int32)
    C = cuda.device_array_like(A)

    

    print(A)

if __name__=='__main__':
    main()