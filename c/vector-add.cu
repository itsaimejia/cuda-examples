#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void vectorAddCPU(int* a, int* b, int* c, int n){
    for(int i=0; i<n; i++){
        c[i]= a[i] + b[i];
    }
}

__global__ void vectorAddGPU(int* a, int* b, int* c, int N){
    int id= blockIdx.x;
    c[id]= a[id] + b[id];
}

int main(){

    //N elementos
    int N = 10;

    //variables para los vectores del host (CPU)
    int *hostA, *hostB, *hostC;

    //variables para los vectores del device (GPU)
    int *deviceA, *deviceB, *deviceC;

    //bytes para los elementos de cada vector
    size_t bytes = sizeof(int) * N;

    //reserva de la memoria para cada vector del Host
    hostA= (int*) malloc(bytes);
    hostB= (int*) malloc(bytes);
    hostC= (int*) malloc(bytes);

    //reserva de la memoria para cada vector del Device
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    //inicializacion de los vectores  A y B del host
    for(int i=0; i<N; i++){
        hostA[i] = 1;
        hostB[i] = 1;
    }

    //copia de la memoria de los vectores del host 
    //hacia los vectores del device
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);

    //definicion de los bloque e hilos para el kernel
    int nThreads = 1;
    int nBlocks = 10; 

    //llamada al metodo e inicializacion del kernel
    vectorAddGPU<<<nBlocks, nThreads>>>(deviceA, deviceB, deviceC, N);

    //copia de la memoria del vector resultado del device
    //hacia el vector del resiltado del host
    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);

    //impresion de resultados
    for(int i=0; i<N; i++){
       printf("%d\t",hostC[i]);
    }

    //liberacion de la memoria de los vectores del device
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}