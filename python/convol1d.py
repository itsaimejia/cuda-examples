import numpy as np
import random as rnd
from numba import cuda


@cuda.jit
def convolution1D(init, mask, result):
    #calcular el id de los hilos
    #otra manera de escribir: 
    # idX = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #con cuda numba es
    idX = cuda.grid(1)
    #calcular la posicion central de la mascara
    c = len(mask) // 2
    #calcular el inicio (start) para el recorrido de la mascara
    start = idX - c 
    #variable temporal para almacenar la suma de productos
    temp = 0
    #uso de for para recorrer la mascara
    for i in range(len(mask)):
        #con ayuda de start ingnorar los valores antes de
        #la posicion central de la mascara al igual que no supere
        #el limite del rango del vector inicial
        if (start + i) >= 0 and (start + i) < len(init):
            #almacena la suma temporal
            temp += init[start + i] * mask[i]
    #asignar el resultado de la suma a 
    # la posicion correspondiente
    result[idX] = temp

def main():
    #definir tamanio de los vectores inicial y resultado 
    N = 10000
    #definir el tamanio de la mascara 
    M = 9

    #crear vector inicial e inicializar con numeros random entre 0 y 100
    memInit = np.asarray([rnd.randrange(100) for _ in range(N)], dtype=np.int32)
    # se almacena en memoria del device
    vectorInit = cuda.to_device(memInit)

    #crear vector mascara e inicializar con numeros random entre 0 y 10
    memMask = np.asarray([rnd.randrange(10) for _ in range(M)], dtype=np.int32)
    # se almacena en memoria del device
    vectorMask = cuda.to_device(memMask)

    #crear un vector resultado del mismo tamanio que el vector A (o B)
    result = cuda.device_array_like(vectorInit)

    #definir numero de hilos por bloque
    nThreads = 10
    #calcular el numero de bloques para el kernel
    nBlocks = N // nThreads

    #instancia metodo e inicializar el kernel
    #pasar por parametro los vectores a sumar y el vector resultado
    convolution1D[nBlocks, nThreads](vectorInit,vectorMask,result)
    
    #copiar la memoria del vector resultado del device a 
    # una variable del host
    res=result.copy_to_host()

    #imprimir los primeros y ultimos 10 valores del vector solucion
    #y el tamanio del mismo
    print('Primeros 10:\n', res[:10])
    print('Ultimos 10:\n', res[-10:])
    print('Tamanio del vector resultado: ', len(res))

if __name__=='__main__':
    main()