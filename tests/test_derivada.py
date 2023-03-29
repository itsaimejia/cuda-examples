import cv2
import numpy as np
import os

file_name = os.path.join(os.path.dirname(__file__), 'avengers.jpg')
assert os.path.exists(file_name)
# Lee la imagen en escala de grises
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

# Calcula la derivada en la dirección horizontal
dx = np.gradient(img)[1]

# Normaliza la magnitud a un rango de 0 a 255
dx = cv2.normalize(dx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Muestra la imagen resultante
cv2.imwrite('derivada_simple.png', dx)
cv2.waitKey(0)
cv2.destroyAllWindows()
