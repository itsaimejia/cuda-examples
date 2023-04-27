import cv2
import os

def main():
    
    #cargar archivo
    file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')

    #leer imagen a color
    img = cv2.imread(file_name)

    #lista de niveles de la piramide
    pyramid = []
    #agregar imagen origen al nivel 0
    pyramid.append(img)

    #crear 2 niveles de imagenes 
    for i in range(2):
        #crear una matriz para cada imagen en la GPU
        img_cuda = cv2.cuda_GpuMat(pyramid[i])
        #crear el filtro gaussianBlur
        blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0)
        #aplicar el filtro a la imagen actual
        apply_blur = blur.apply(img_cuda)
        #redimensionar la imagen 
        img_down = cv2.cuda.pyrDown(apply_blur)
        #descargar imagen resultado
        r_img_down = img_down.download()
        #agregar a la piramide
        pyramid.append(r_img_down)
    
    #recorrer la lista de la piramide para crear las imagenes de cada nivel
    for i in range(len(pyramid)):
        cv2.imwrite('lvl-cuda-{}.png'.format(i), pyramid[i])

    
if __name__=='__main__':
    main()



