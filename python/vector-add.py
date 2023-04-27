import numpy as np
from numba import cuda

@cuda.jit
def vectorAddGPU(a, b, c):
    #calcular el id de los hilos
    idX = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if(idX<len(c)):
        c[idX] = a[idX] + b[idX]

def main():
    #definir tamanio de los vectores
    N = 10000
    
    #crear vectores A y B de 1's de tipo entero 
    #y se almacena en memoria del device
    A = cuda.to_device(np.ones(N, dtype=np.int32))
    B = cuda.to_device(np.ones(N, dtype=np.int32))
    #crear un vector resultado del mismo tamanio que el vector A (o B)
    C = cuda.device_array_like(A)

    #definir numero de hilos por bloque
    nThreads = 10
    #calcular el numero de bloques para el kernel
    nBlocks = N // nThreads

    #instancia metodo e inicializar el kernel
    #pasar por parametro los vectores a sumar y el vector resultado
    vectorAddGPU[nBlocks, nThreads](A,B,C)
    
    #copiar la memoria del vector resultado del device a 
    # una variable del host
    c=C.copy_to_host()

    #imprimir los primeros y ultimos 10 valores del vector solucion
    #y el tamanio del mismo
    print(c[:10], c[-10:], len(c))

if __name__=='__main__':
    main()