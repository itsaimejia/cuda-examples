import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image to be detected
img = cv2.imread('test2.jpg')

blur = cv2.GaussianBlur(img,(5,5),0)

# Convert the image to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()