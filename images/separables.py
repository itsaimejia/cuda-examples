import numpy as np
import cv2
import os
from lib_kernels import convolve2D
import time 

#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
assert os.path.exists(file_name)

#leer imagen
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#crear imagen escala de grises
cv2.imwrite('grey.png', img)

#definir kernel sobel (3x3)
kernel_sobel = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])
#factorizar
#kernel 1x3
kernel_sobel1x3 = np.array([[-1,0,1]])
#kernel 3x1
kernel_sobel3x1 = np.array([[1],[2],[1]])

#aplicar convolucion normal, kernel 3x3
sobel3x3 = convolve2D(img,kernel_sobel)

#aplicar convolucion kernel separados
#con kernel 1x3
sobel1x3 = convolve2D(img,kernel_sobel1x3)
#convolucion al resultado obtenido con kernel 3x1
sobel_sep = convolve2D(sobel1x3,kernel_sobel3x1)

#imagenes resultado
cv2.imwrite('sobel3x3.png',sobel3x3 )
cv2.imwrite('sobel_sep.png',sobel_sep )