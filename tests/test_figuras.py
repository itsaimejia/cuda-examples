import cv2
import os

for i in range(4):
    #cargar una a una las imagenes
    file_name = os.path.join(os.path.dirname(__file__), 'figuras{}.png'.format(i+1))
    print('Imagen: figuras{}.png'.format(i+1))
    #leer cada imagen
    img = cv2.imread(file_name)
    #convertir cada imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Escala de grises', gray)
    #por el metodo canny se detectan los bordes 
    #definir el rango de intensidad de cada pixel para tomar en cuenta de la escala de grises
    #entre 50 y 150 (la intensidad puede variar de 0 a 255)
    edges = cv2.Canny(gray, 100, 150)
    cv2.imshow('Bordes detectados', edges)
    #encontrar los contornos
    #con cv2.RETR_EXTERNAL se indica que se obtengan los contornos externos
    #con cv2.CHAIN_APPROX_SIMPLE se indica que la aproximacion sea simple
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #iterar sobre los contornos encontrados en la imagen
    for cnt in contours:
        #aproximar los contornos de las figuras geometricas
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        #detectar si la figura tiene 3 lados (es un triangulo)
        if len(approx) == 3:
            #dibujar la figura geometrica en la imagen original en base 
            #la aproximacion de contornos de color verde 
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
            print('Triángulo')
            #-----aqui se enciende el led----
        #detectar figuras con 4 lados
        elif len(approx) == 4:
            #obtiene las coordenadas x e y, asi como el valor del ancho y alto (w, h)
            #de la figura
            x, y, w, h = cv2.boundingRect(approx)
            #con el ancho y alto se obtiene el ratio para determinar si es un cuadrado o un rectangulo
            aspect_ratio = float(w) / h
            #el ratio debe ser de 1, si se mantiene en un rango de 0.05 de margen de error
            #es un cuadrado
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                #dibujar la figura geometrica en la imagen original en base 
                #la aproximacion de contornos de color azul
                cv2.drawContours(img, [approx], 0, (255, 0, 0), 3)
                print('Cuadrado')
                #-----aqui se enciende el led----

    # Muestra la imagen con las figuras detectadas
    cv2.imshow('Figuras detectadas', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()