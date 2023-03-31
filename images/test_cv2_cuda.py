import numpy as np
import os
import cv2
import time

    
def main():
    
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
    assert os.path.exists(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    start = time.time()
    src = cv2.cuda_GpuMat()
    src.upload(img)
    gray = cv2.cuda.cvtColor(src,cv2.COLOR_RGB2GRAY)
    src_sobel_x = cv2.cuda.createSobelFilter(cv2.CV_8U,cv2.CV_8U,1, 0,3)
    dts_sobel_x = src_sobel_x.apply(gray)

    end = time.time()
    dst = dts_sobel_x.download()

    print(end - start)

    cv2.imwrite('sobel.png',dst)

    
if __name__=='__main__':
    main()