import cv2
import os

src_video =  os.path.join(os.path.dirname(__file__), 'gatos.mp4')
src_classifier = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalcatface_extended.xml')
cap = cv2.VideoCapture(src_video)
face_cascade = cv2.CascadeClassifier(src_classifier)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    # Realizar operaciones en el cuadro de video (por ejemplo, mostrarlo en una ventana, procesarlo, etc.)
    blur = cv2.GaussianBlur(frame,(5,5),0)
    # Convert the image to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Mostrar el cuadro de video en una ventana
    cv2.imshow('Video', frame)
    # Esperar por un milisegundo y salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar el objeto de captura y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
