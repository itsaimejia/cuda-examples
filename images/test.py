import numpy as np
import cv2
import os
import math 
import time
from numba import jit

@jit
def to_grey(img):
    width = img.shape[0]
    height = img.shape[1]
    res = np.zeros((width,height), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):     
            res[i, j] = math.floor(sum(img[i, j]) // 3)
    return res

@jit
def convolve2D(image, kernel):
    # Dimensiones de la imagen y el kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    
    # Inicializar la matriz de salida
    output = np.zeros(image.shape)
    
    # Calcular el padding
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    
    # Aplicar el padding a la imagen
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    
    # Realizar la convolución
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = (kernel * padded_image[row:row + kernel_row, col:col + kernel_col]).sum()
            
    return output

file_name = os.path.join(os.path.dirname(__file__), 'avengers.jpg')
assert os.path.exists(file_name)

img = cv2.imread(file_name)
start_grey = time.time()
img2 = to_grey(img)
end_grey = time.time()
print('Time grey: ', end_grey-start_grey)
mask = np.array([[1,1,1],[1,8,1],[1,1,1]])
start_convolve = time.time()
res = convolve2D(img2, mask)
end_convolve = time.time()
print('Time convolve: ', end_convolve - start_convolve)

# write image
cv2.imwrite("output.png", res)
cv2.waitKey(0)



