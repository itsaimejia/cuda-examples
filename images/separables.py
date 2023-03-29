import numpy as np
import cv2
import os
from lib_kernels import convolve2D
import time 

file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
assert os.path.exists(file_name)

img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('grey.png', img)
kernel_gauss5x5 = np.multiply((1/273),np.array([[1,4,7,4,1],
                                                [4,16,26,16,4],
                                                [7,26,41,26,7],
                                                [4,16,26,16,4],
                                                [1,4,7,4,1]]))

kernel_gauss1x5 = np.multiply((1/273),np.array([[1,4,7,4,1]]))
kernel_gauss5x1 = np.multiply((1/273),np.array([[1],[4],[7],[4],[1]]))


start5x5 = time.time()
gauss5x5 = convolve2D(img,kernel_gauss5x5)
end5x5 = time.time()

start_sep = time.time()
gauss5x1 = convolve2D(img,kernel_gauss5x1)
gauss_sep = convolve2D(gauss5x1,kernel_gauss1x5)
end_sep= time.time()

print(end5x5-start5x5)
print(end_sep-start_sep)
cv2.imwrite('gauss5x5.png',gauss5x5 )
cv2.imwrite('gauss_sep.png',gauss_sep )