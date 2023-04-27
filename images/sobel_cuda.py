import cv2
import numpy as np
import os
import inspect

def main():
     #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
    #leer imagen a color
    img = cv2.imread(file_name)
    #crear una matriz de GPU
    c_img = cv2.cuda_GpuMat()
    #cargar la imagen en la memoria de la gpu
    c_img.upload(img)
    #convertir imagen a escala de grises
    c_gray = cv2.cuda.cvtColor(c_img, cv2.COLOR_RGB2GRAY)

    #crear un filtro (gradiente) para el eje x
    sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_32F, 1, 0)
    #crear un filtro (gradiente) para el eje y
    sobel_y = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_32F, 0, 1  )
    
    #aplicar ambos filtros por separado a la imagen
    grad_x_cuda = sobel_x.apply(c_gray)
    grad_y_cuda = sobel_y.apply(c_gray)

    #obtener la memoria de la gpu en una variable local
    #para cada gradiente
    grad_x = grad_x_cuda.download()
    grad_y = grad_y_cuda.download()

    #calcular la magnitud 
    mag = cv2.magnitude(grad_x, grad_y)

    #crear imagenes
    cv2.imwrite('sobel_x.png', grad_x)
    cv2.imwrite('sobel_y.png', grad_y)
    cv2.imwrite('sobel_cv2.png', mag)

    
if __name__=='__main__':
    main()