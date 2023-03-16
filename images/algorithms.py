
import numpy as np
import cv2
import os
from lib.filter2d import apply_kernel


def main():
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
    assert os.path.exists(file_name)

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    
    img_result=apply_kernel(img,'prewitt_y')
    # Muestra la imagen resultante
    cv2.imwrite('result.png', img_result)

if __name__=='__main__':
    main()