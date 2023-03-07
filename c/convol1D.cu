#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/*
params
init: vector inicial
mask: vector mascara
result: vector para resultado
n: numero de elementos del vector inicial
m: numero de elementos de la mascara 
*/
__global__ void convolution1D(int *init, int *mask, int *result, int n, int m){
    //calcular el id de cada hilo 
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    //calcular la posicion central de la mascara
    int c = m / 2;

    //calcular el start de la mascara 
    int start = id - c;

    //variable para almacenar la suma de los productos 
    int temp = 0;

    //recorrido de la mascara 
    for(int i=0; i<m; i++){
        //verificar con ayuda de start que ignore todos los valores 
        //previos a la posicion central de la mascara y que no 
        //supere el final del vector inicial 
        if(((start + i) >= 0) && ((start + i) < n )){
            //almacenar el acomulado de la suma
            temp += init[start + i] * mask[i];
        }
    }
    //asignar el valor de la suma temporal a cada posicion del vector resultado
    result[id] = temp;

}
int main(){
    //numero de elementos en el vector inicial y resultado
    int n = 10000;
    //tamanio de la mascara
    int m = 9;

    //bytes para los elementos del vector inicial, resultado y la mascara
    size_t bytesN = sizeof(int) * n;
    size_t bytesM = sizeof(int) * m;

    //variables para el vector inicial, la mascara y el vector resultad del host (CPU)
    int *hostInit, *hostMask, *hostResult;

    //variables para el vector inicial, la mascara y el vector resultado del device (GPU)
    int *deviceInit, *deviceMask, *deviceResult;

    //reserva de la memoria para vector, mascara y resultado del Host
    hostInit= (int*) malloc(bytesN);
    hostMask= (int*) malloc(bytesM);
    hostResult = (int*) malloc(bytesN);

    //reserva de la memoria para vector, mascara y resultado del Device
    cudaMalloc(&deviceInit, bytesN);
    cudaMalloc(&deviceMask, bytesM);
    cudaMalloc(&deviceResult, bytesN);

    //inicializar vector inicial con valores aleatorios entre 0 y 100
    for(int i=0; i<n; i++){
        hostInit[i] = rand() % 100;
    }

    //inicializar mascara con valores aleatorios entrre 0 y 10
    for(int j=0; j<m; j++){
        hostMask[j] = rand() % 10;
    }

    //copia de la memoria del vector y la mascara del host al device
    cudaMemcpy(deviceInit, hostInit, bytesN, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMask, hostMask, bytesM, cudaMemcpyHostToDevice);

    //hilos por bloque
    int nHilos = 128;
    
    //numero de bloques 
    int nBloques = (int) ceil((n + nHilos) / nHilos); 

    //inicializacion del kernel y ejecucion del metodo 
    convolution1D<<<nBloques, nHilos>>>(deviceInit, deviceMask, deviceResult, n, m);

    //copia de la memoria del vector resultado del device al host
    cudaMemcpy(hostResult, deviceResult, bytesN, cudaMemcpyDeviceToHost);

    //impresion de resultados
    for(int i=0; i<n; i++){
        printf("|%i - %d|\t",i,hostResult[i]);       
    }

    //liberacion de la memoria de los vectores
    cudaFree(deviceInit);
    cudaFree(deviceResult);
    cudaFree(deviceMask);
    
}