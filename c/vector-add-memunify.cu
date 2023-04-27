#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAddGPU(int* a, int* b, int* c, int N){
    int id= blockIdx.x * blockDim.x + threadIdx.x;
    if(id<N){
        c[id]= a[id] + b[id];
    }
}

int main(){

    //N elementos
    int N = 10000;

    //variables para los vectores del device (GPU)
    int *deviceA, *deviceB, *deviceC;

    //bytes para los elementos de cada vector
    size_t bytes = sizeof(int) * N;

    //reserva de la memoria para cada vector del Device
    cudaMallocManaged(&deviceA, bytes);
    cudaMallocManaged(&deviceB, bytes);
    cudaMallocManaged(&deviceC, bytes);

    //inicializacion de los vectores  A y B del host
    for(int i=0; i<N; i++){
        deviceA[i] = 3;
        deviceB[i] = 1;
    }

    //definicion de los bloque e hilos para el kernel
    int nThreads = 100;
    int nBlocks = (int) ceil(N / nThreads); 

    //llamada al metodo e inicializacion del kernel
    vectorAddGPU<<<nBlocks, nThreads>>>(deviceA, deviceB, deviceC, N);

    cudaDeviceSynchronize();
    //impresion de resultados
    for(int i=0; i<N; i++){
       printf("%i - %d\t",i,deviceC[i]);
    }

    return 0;
}