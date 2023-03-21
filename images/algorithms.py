
import numpy as np
import cv2
import os
import time
from lib.filter2d import apply_kernel


def main():
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
    assert os.path.exists(file_name)

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    
    
    start = time.time()
    img_result = apply_kernel(img)
    end = time.time()
    # Muestra la imagen resultante
    print('tiempo:',end - start)
    # Muestra la imagen resultante
    cv2.imwrite('result.png', img_result)

if __name__=='__main__':
    main()