import numpy as np
from numba import cuda

'''
params:
init: matriz inicial
mask: matriz mascara
result: matriz resultado
'''
@cuda.jit
def convolution2D(init, mask, result):

    # #calcular el id de los hilos para los ejes X (columna) e Y(renglon)
    # colId = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # rowId = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    colId, rowId = cuda.grid(2)
   
    #tamanio de un eje de la mascara 
    m = mask.shape[0]
    #calcular ancho o largo de la matriz resultado
    n = result.shape[0]
    #calcular la mitad de la dimension de la mascara
    c = m // 2
    #calcular la posicion central de la mascara (start)
    startRow = rowId - c
    startCol = colId - c

    #variable temporal para la suma de los productos
    temp = 0

    #uso de for para recorrer la mascara
    #iterar sobre los renglones 
    for i in range(m):
        #iterar sobre las columnas
        for j in range(m):
            #verificar que se ignoren los valores previos a
            #la posicion central en los 2 ejes y que esta no 
            #supere el rango de los renglones de la matriz inicial
            if int(startRow + i) >= 0 and int(startRow + i) < n:
                #revisar que no supere el limite de las columnas
                if int(startCol + j) >= 0 and int(startCol + j) < n:
                    temp+= init[int(startRow + i),int( startCol + j)] * mask[i,j]
    
    result[rowId,colId] = temp
        


def main():
    #dimension de la matriz inicial cuadrada (N * N)  
    N = 100
    #dimension de la mascara (M * M)
    M = 9

    #crear matriz init de N * N con todas las posiciones inicializadas en 2
    #crear matriz mask de M * M de 1's
    #copiar la memoria de las matrices del host al device
    init = cuda.to_device(np.full((N, N), 2, dtype=np.int32))
    mask = cuda.to_device(np.ones([M,M], dtype=np.int32))
    #crear una matriz resultado de dimension N * N con 0's
    result = cuda.to_device(np.zeros([N,N]))

    #definir los hilos por bloque y calcular el numero de bloques
    nThreads = 10
    nBlocks = (N + nThreads - 1) // nThreads
    #definir una tupla para los hilos por bloque
    threadsPerBlock = (nThreads, nThreads)
    #definir una tupla para los bloques por malla/cuadricula
    blocksPerGrid = (nBlocks, nBlocks)

    #instanciar el metodo e inicializar el kernel
    convolution2D[blocksPerGrid, threadsPerBlock](init,mask,result)

    #copiar al host la memoria de la matriz resultado
    res = result.copy_to_host()
    
    #imprimir los primeros y ultimos 10 valores de la matriz solucion
    #y el tamanio de la misma
    print('Primeros 10:\n', res[0,:100])
    print('Intemedio: \n', res[N//2,:100])
    print('Ultimos 10:\n',res[N-1,-100:])
    print('Tamanio matriz:', res.shape[0] * res.shape[1])

if __name__=='__main__':
    main()