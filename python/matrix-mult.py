
import numpy as np
from numba import cuda

'''
params:
a: matriz A
b: matriz B
c: matriz resultado
'''
@cuda.jit
def matrixMult(a, b, c):
    #calcular el id de los hilos para los ejes X (columna) e Y(renglon)
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    #calcular ancho o largo de la matriz resultado
    n = c.shape[0]

    #evitar que no se utilicen mas hilos que la cantidad de elementos 
    #por renglon o columna
    if row < n and col < n:
        #variable temporal para almacenar la suma 
        temp = 0
        #iterar las matrices sobre los renglones y columnas 
        for i in range(a.shape[0]):
                temp += a[row,i] * b[i,col]
        c[row,col] = temp


def main():
    #definir la dimension de la matriz cuadrada (N * N)  
    N = 1000

    #crear matrices A y B de N * N tamanio de 1's de tipo entero
    #copiar la memoria de las matrices del host al device
    A = cuda.to_device(np.ones([N,N], dtype=np.int32))
    B = cuda.to_device(np.ones([N,N], dtype=np.int32))
    #crear una matriz resultado de dimension N * N con 0's
    C = cuda.to_device(np.zeros([N,N]))

    #definir los hilos por bloque y calcular el numero de bloques
    nThreads = 10
    nBlocks = N // nThreads
    #definir una tupla para los hilos por bloque
    threadsPerBlock = (nThreads, nThreads)
    #definir una tupla para los bloques por malla/cuadricula
    blocksPerGrid = (nBlocks, nBlocks)

    #instanciar el metodo e inicializar el kernel
    matrixMult[blocksPerGrid, threadsPerBlock](A,B,C)

    #copiar al host la memoria de la matriz resultado
    c = C.copy_to_host()
    
    #imprimir los primeros y ultimos 10 valores de la matriz solucion
    #y el tamanio de la misma
    print('Primeros 10:\n',c[:10])
    print('Ultimos 10:\n',c[-10:])
    print('Tamanio matriz:', c.shape[0] * c.shape[1])

if __name__=='__main__':
    main()