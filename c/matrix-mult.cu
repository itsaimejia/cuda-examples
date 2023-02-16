#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

__global__ void matrixMult(int* a, int* b, int* c, int n){
    //calcular el id de cada hilo de cada renglon
    int renglon = blockIdx.y * blockDim.y + threadIdx.y;
    //calcula el id de cada hilo de cada columna
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    //variable para almacenar la suma temporal de cada renglon
    int suma_temporal= 0;

    //evitar que no se utilicen mas hilos que la cantidad de elementos 
    //por renglon o columna
    if((renglon<n) && (columna<n)){
        //iterar las matrices sobre los renglones y columnas 
        for(int it=0; it<n; it++){
            //guardar el valor de la suma de la multiplicacion de cada elemento
            suma_temporal += a[renglon * n + it] * b[it * n + columna];
        }
        //asignamos el valor de la suma temporal a la matriz solucion
        c[renglon * n + columna]= suma_temporal;
    }
}

int main(){
    //dimension matriz N * N (1000 * 1000)
    int N = 10;

    //bytes para los elementos de cada matriz
    size_t bytes = sizeof(int) * N * N;

    //variables para las matrices del host (CPU)
    int *hostA, *hostB, *hostC;

    //variables para las matrices del device (GPU)
    int *deviceA, *deviceB, *deviceC;

    //reserva de la memoria para cada matriz del Host
    hostA= (int*) malloc(bytes);
    hostB= (int*) malloc(bytes);
    hostC= (int*) malloc(bytes);

    //reserva de la memoria para cada matriz del Device
    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    //inicializacion de las matices A y B del host
    for(int renglon=0; renglon<N; renglon++){
        for(int columna=0; columna<N; columna++){
            hostA[renglon * N + columna] = 1;
            hostB[renglon * N + columna] = 1;
        }
        
    }

    //copia de la memoria de las matrices del host al device
    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);

    //hilos por bloque
    int nHilos = 10;
    
    //bloques en cada cuadricula (dimension)
    int nBloques = (int) ceil(N / nHilos); 

    //variables de 3 dimensiones 
    dim3 cuadricula(nBloques, nBloques); //cuadricula(100, 100, 1); /malla/grid
    dim3 hilos(nHilos, nHilos); // hilos(10, 10, 1);

    //inicializacion del kernel y ejecucion del metodo 
    matrixMult<<<cuadricula, hilos>>>(deviceA, deviceB, deviceC, N);

    //copia de la memoria de la matriz resultado del device al host
    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);

    //impresion de resultados
    for(int i=0; i<N; i++){
       for(int j=0; j<N; j++){
        printf("%d\t",hostC[i * N + j]);
       }
       
    }

    //liberacion de la memoria de los vectores del device
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

}