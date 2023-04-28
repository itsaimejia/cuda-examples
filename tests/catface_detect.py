import cv2
import os

src_video =  os.path.join(os.path.dirname(__file__), 'gatos.mp4')
src_classifier = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalcatface_extended.xml')
print(src_classifier)
cap = cv2.VideoCapture(src_video)
face_cascade = cv2.cuda.CascadeClassifier_create(src_classifier)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    frame_cuda = cv2.cuda_GpuMat(frame)
    # Realizar operaciones en el cuadro de video (por ejemplo, mostrarlo en una ventana, procesarlo, etc.)
    blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)
    apply_blur = blur.apply(img_cuda)
    # Convert the image to grayscale
    gray = cv2.cuda.cvtColor(apply_blur, cv2.COLOR_RGB2GRAY)
    # Detect faces in the grayscale image
    faces_detect = face_cascade.detectMultiScale(gray)
    faces = faces_detect.download()
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