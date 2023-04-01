import cv2
import os

def main():
    
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')

    #leer imagen a color
    img = cv2.imread(file_name)

    pyramid = []

    pyramid.append(img)

    for i in range(2):
        img_cuda = cv2.cuda_GpuMat(pyramid[i])
        blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0)
        apply_blur = blur.apply(img_cuda)
        img_down = cv2.cuda.pyrDown(apply_blur)
        r_img_down = img_down.download()
        pyramid.append(r_img_down)
    
    for i in range(len(pyramid)):
        cv2.imwrite('lvl-cuda-{}.png'.format(i), pyramid[i])

    
if __name__=='__main__':
    main()



