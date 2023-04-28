import cv2
import os
import Jetson.GPIO as GPIO

#cargar video
src_video = os.path.join(os.path.dirname(__file__), 'gatos.mp4')
#leer video 
cap = cv2.VideoCapture(src_video)

#cargar archivo de eentrenamiento
src_classifier = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalcatface_extended.xml')
#crear clasificador
face_cascade = cv2.CascadeClassifier(src_classifier)

#modo de enumeracion de los pines 
GPIO.setmode(GPIO.BOARD)
#numeros de pines
pinGreen = 12
pinRed = 11
#configuracion de pines 
GPIO.setup(pinGreen, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(pinRed, GPIO.OUT, initial=GPIO.LOW)
high = GPIO.HIGH
low = GPIO.LOW
#ciclo infinito
while True:
    #leer video cuadro por cuadro
    ret, frame = cap.read()
    if not ret:
        break
    #aplicar filtro gaussiano al cuadro 
    blur = cv2.GaussianBlur(frame,(5,5),0)
    #convertir a escala de grises el cuadro
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #detectar rostros de gatos en base al entrenador
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #si hay rostros ilumina el led verde
    #en caso contrario el led rojo
    if (len(faces) > 0):
        GPIO.output(pinGreen, high)
        GPIO.output(pinRed, low)
    else:
        GPIO.output(pinRed, high)
        GPIO.output(pinGreen, low)
    #dibujar cuadrados en los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # mostrar cada cuadro en una ventana
    cv2.imshow('Video', frame)
    #cerrar cuando termine el video o cuando presione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#liberar el objeto de captura y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
