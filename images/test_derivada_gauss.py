import cv2
import numpy as np
import os

def derivada_gaussiana(img):
    # Aplica un filtro de suavizado
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Calcula las derivadas parciales en la dirección horizontal y vertical
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # Calcula la magnitud y dirección de la gradiente
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)

    # Normaliza la magnitud a un rango de 0 a 255
    return cv2.normalize(mag, ang, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

file_name = os.path.join(os.path.dirname(__file__), 'avengers.jpg')
assert os.path.exists(file_name)
# Lee la imagen en escala de grises
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

result = derivada_gaussiana(img)

# Muestra la imagen resultante
cv2.imwrite('derivada_gaussiana.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()