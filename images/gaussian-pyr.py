import cv2
import os


#cargar archivo
file_name = os.path.join(os.path.dirname(__file__), 'hamster.jpg')
#leer imagen a color
img = cv2.imread(file_name)
#lista para almacenar las imagenes de cada nivel de la piramide
pyramid = []
#se agrega la imagen original a la lista (nivel 0)
pyramid.append(img)
#el ciclo for esta definido en 2, por lo que se obtendran 2 imagenes 
#de la piramide
for i in range(2):
    #aplicar filtro gaussiano 3x3 a cada imagen (empieza por el nivel 0)
    img = cv2.GaussianBlur(pyramid[i],(3,3),0)
    #reducir la resolucion con el metodo pyrDown(src)
    img = cv2.pyrDown(img)
    #agregar la imagen resultado a la lista
    pyramid.append(img)

#crear las imagenes de todos los niveles
for i in range(len(pyramid)):
    cv2.imwrite('lvl-{}.png'.format(i), pyramid[i])