import numpy as np
import os
import cv2
import time

    
def main():
    
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')

    #leer imagen a color
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    start = time.time()
    #crear una matriz de GPU
    c_img = cv2.cuda_GpuMat()
    #cargar la imagen en la memoria de la gpu
    c_img.upload(img)
    #convertir imagen a escala de grises
    c_gray = cv2.cuda.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
    #crear filtro (crear kernel)
    c_filter= cv2.cuda.createGaussianFilter(cv2.CV_8U,cv2.CV_8U,(3,3),0)
    #aplicar filtro a la imagen 
    c_result = c_filter.apply(c_gray)
    
    #obtener la memoria de la en una variable local 
    result = c_result.download()
   
    end = time.time()
    print('Con opencv+cuda: ', end - start)
    #crear imagen resultado
    cv2.imwrite('gaussian_cv2.png',result)

    
    
if __name__=='__main__':
    main()