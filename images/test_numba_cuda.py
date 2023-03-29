import numpy as np
from numba import cuda
import cv2
import os
from lib_kernels import sharpen, edge, sobel, prewitt, gaussianBlur, convolve2D
import time 

@cuda.jit
def convolve2D(img, kernel):

    #calcular el id de los hilos para los ejes X (columna) e Y(renglon)
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    #dimensiones imagen origen y kernel
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    

    #definir el tamaño del relleno para cada eje 
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    temp += 0
    #crear una matriz de 0's con el relleno agregado
    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    #agregar la imagen origen a la matriz de 0's con 
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img
    
    #calcular cada nuevo pixel multiplicando la cada valor del kernel con
    # #la seccion de la matriz imagen concidente
    pixel = np.multiply(kernel, padded_img[row:row + kernel_row, col:col + kernel_col])

    for i in range(kernel_row):
        #iterar sobre las columnas
        for j in range(kernel_col):
            #verificar que se ignoren los valores previos a
            #la posicion central en los 2 ejes y que no 
            #supere el rango de los renglones de la matriz inicial
            if row < img_row:
                #revisar que no supere el limite de las columnas
                if col < img_col:
                    temp+= padded_img[row,col] * kernel[i,j]
    #sumar los productos y asignar a la posicion correspondiente de la matriz resultado
    img_res[row, col] = temp
    
   

#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
assert os.path.exists(file_name)


img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
img_row, img_col = img.shape
img_res = np.zeros(img.shape)
kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125,0.25,0.125],
                       [0.0625, 0.125, 0.0625]])
device_img = cuda.to_device(np.array(img))
device_kernel = cuda.to_device(kernel)
nThreads = 4
nBlocksRow = (img_row + nThreads -1) // nThreads
nBlocksCol = (img_col + nThreads -1) // nThreads
#definir una tupla para los hilos por bloque
threadsPerBlock = (nThreads, nThreads)
#definir una tupla para los bloques por malla/cuadricula
blocksPerGrid = (nBlocksRow, nBlocksCol)
#instanciar el metodo e inicializar el kernel
convolve2D[blocksPerGrid, threadsPerBlock](img,kernel,img_res)
#copiar al host la memoria de la matriz resultado
res = img_res.copy_to_host()

print(res)

# cv2.imwrite('grey.png', img)
# img_result = sobel(img, 'x')

# cv2.imwrite('result.png',img_result)