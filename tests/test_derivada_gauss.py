import cv2
import numpy as np
import os
from numba import jit
import time

@jit(forceobj=True)
def derivada_gaussiana(img):
    # Aplica un filtro de suavizado
   return cv2.GaussianBlur(img, (3, 3), 0)


    # # Calcula las derivadas parciales en la dirección horizontal y vertical
    # dx = cv2.Sobel(img2, cv2.CV_64F, 1, 0)
    # dy = cv2.Sobel(img2, cv2.CV_64F, 0, 1)

    # # Calcula la magnitud y dirección de la gradiente
    # mag = np.sqrt(dx**2 + dy**2)
    # ang = np.arctan2(dy, dx)

    # # Normaliza la magnitud a un rango de 0 a 255
    #  cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
assert os.path.exists(file_name)
# Lee la imagen en escala de grises
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

start = time.time()
result = derivada_gaussiana(img)
end = time.time()
# Muestra la imagen resultante
print('tiempo:',end - start)
cv2.imwrite('derivada_gaussiana.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()