from numba import jit
import numpy as np
import cv2
import os
import math 
import time

@jit
def img_to_grey(img):
    height, width = img.shape[0:2]
    output = np.zeros(img.shape[0:2])
    for i in range(height):
        for j in range(width):    
            b = img[i,j][0]
            g = img[i,j][1]
            r = img[i,j][2]  
            output[i, j] = math.floor(sum(img[i, j]) / 3)
    
    return output

@jit
def convolve2D(img, mask):
    img_row, img_col = img.shape
    mask_row, mask_col = mask.shape

    img_res = np.zeros(img.shape)

    pad_height = int((mask_row - 1) / 2)
    pad_width = int((mask_col - 1) / 2)

    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img
    
    for row in range(img_row):
        for col in range(img_col):
            img_res[row, col] = (mask * padded_img[row:row + mask_row, col:col + mask_col]).sum()
    
    return img_res


def main():

    file_name = os.path.join(os.path.dirname(__file__), 'gato.png')
    assert os.path.exists(file_name)

    img = cv2.imread(file_name)
    
    start_grey = round(time.time())
    img_grey = img_to_grey(img)
    end_grey = round(time.time())
    print('Escala de grises tomo: ', end_grey - start_grey, ' segundos')
    cv2.imwrite("escala_grises.png", img_grey)
    
    mask = np.array([[1, 1, 1], [1, 3, 1], [1, 1, 1]])
    start_convol = round(time.time())
    img_convol = convolve2D(img_grey, mask)
    end_convol = round(time.time())
    print('Convolucion tomo: ', end_convol - start_convol, ' segundos')
    cv2.imwrite("convolucion_2d.png", img_convol)
    
    print('Ambos procesos tomaron:', (end_grey - start_grey) + (end_convol - start_convol), ' segundos')
    

if __name__=='__main__':
    main()