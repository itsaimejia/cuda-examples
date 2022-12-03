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
    int id= blockIdx.x * blockDim.x+threadIdx.x;
    if(id<N){
        c[id]= a[id] + b[id];
    }
}

int main(){

    
    int N = 10000;

    int *hostA, *hostB, *hostC;

    int *deviceA, *deviceB, *deviceC;

    size_t bytes = sizeof(int) * N;

    hostA= (int*) malloc(bytes);
    hostB= (int*) malloc(bytes);
    hostC= (int*) malloc(bytes);

    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    for(int i=0; i<n; i++){
        hostA[i] = i;
        hostB[i] = i;
    }

    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);

    int nThreads = 128;
    int nBlocks = (int)ceil(N / nThreads); 

    vectorAddGPU<<<nBlocks, nThreads>>>(deviceA, deviceB, deviceC, N);

    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++){
       printf("%d\t",&c[i]);
    }
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}