#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/*
params
init: matriz inicial
mask: matriz mascara
result: matriz para resultado
N: ndimension de la matriz inicial
M: dimension  de la mascara 
*/
__global__ void convolution2D(int *init, int *mask, int *result, int N, int M){
    //calcular el id de los hilos de cada eje 
    int idY = blockIdx.y * blockDim.y + threadIdx.y; //renglon
    int idX = blockIdx.x * blockDim.x + threadIdx.x; //columna
   

    //calcular la mitad de la dimension de la mascara
    int c = M / 2;

    //calcular  posicion central (start) para cada eje de la mascara 
    int startY = idY - c;
    int startX = idX - c;
   

    //variable para almacenar la suma de los productos 
    int temp = 0;

    //recorrer la mascara 
    //iterar sobre los renglones
    for(int i=0; i<M; ++i){
        //iterar sobre las columnas
        for(int j=0; j<M; ++j){
            //verificar con ayuda de start (X/Y) que ignore todos los valores 
            //previos a la posicion central de la mascara y que no 
            //supere el final matriz inicial
            //revisar el rango de los renglones de la matriz inicial 
            if(((startY + i) >= 0) && ((startY + i) < N )){
                //revisar el rango de los renglones de la matriz inicial 
                if(((startX + j) >= 0) && ((startX + j) < N )){
                    //almacenar el acomulado de la suma
                    temp += init[(startY + i) * N + (startX + j)] * mask[i * M + j];
                }
            }
        }
    }
    //asignar el valor de la suma temporal a cada posicion de la matriz resultado
    result[idY * N + idX] = temp;

}
int main(){
    //dimension matriz inicial y resultado N * N (100 * 100)
    int N = 100;
    //dimension de la mascara M * M (9 * 9)
    int M = 9;

    //bytes para los elementos de la matriz inicial, resultado y la mascara
    //dimensiones cuadradas: N*N // M*M
    size_t bytesN = sizeof(int) * N * N;
    size_t bytesM = sizeof(int) * M * M;

    //variables para la matrices: inicial, la mascara y el matriz resultado del host (CPU)
    int *hostInit, *hostMask, *hostResult;

    //variables para las matrices: inicial, la mascara y el matriz resultado del device (GPU)
    int *deviceInit, *deviceMask, *deviceResult;

    //reserva de la memoria para matriz, mascara y resultado del Host
    hostInit= (int*) malloc(bytesN);
    hostMask= (int*) malloc(bytesM);
    hostResult = (int*) malloc(bytesN);

    //reserva de la memoria para matriz, mascara y resultado del Device
    cudaMalloc(&deviceInit, bytesN);
    cudaMalloc(&deviceMask, bytesM);
    cudaMalloc(&deviceResult, bytesN);

    //linearizar ambas matrices 
    //inicializar la matriz inicial con 2
    for(int row=0; row<N; row++){
        for(int col=0; col<N; col++){
            hostInit[row * N + col] = 2;
        }
    }
    //inicializar mascara con 1
    for(int rowM=0; rowM<M; rowM++){
        for(int colM=0; colM<M; colM++){
            hostMask[rowM * M + colM] = 1;
        }
    }

    //copia de la memoria de la matriz y la mascara del host a las matrices del device
    cudaMemcpy(deviceInit, hostInit, bytesN, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, hostMask, bytesM, cudaMemcpyHostToDevice);

    //hilos por bloque
    int nThreads = 10;
    
    //bloques en cada cuadricula (dimension)
    int nBlocks = (N + nThreads - 1)/ nThreads ; 

    //variables de 3 dimensiones 
    dim3 grid(nBlocks, nBlocks); //malla / cuadricula de bloques
    dim3 blocks_dim(nThreads, nThreads); //hilos por bloque

    //inicializacion del kernel y ejecucion del metodo 
    convolution2D<<<grid, blocks_dim>>>(deviceInit, deviceMask, deviceResult, N, M);

    //copia de la memoria de la matriz resultado del device al host
    cudaMemcpy(hostResult, deviceResult, bytesN, cudaMemcpyDeviceToHost);

    //impresion de resultados
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
         printf("[%i][%i]: %d\t",i,j,hostResult[i * N + j]);
        }  
     }
 
    //liberacion de la memoria de los vectores
    cudaFree(deviceInit);
    cudaFree(deviceResult);
    cudaFree(deviceMask);
    
}