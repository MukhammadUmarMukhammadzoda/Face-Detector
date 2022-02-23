import cv2

image = cv2.imread('Images/img3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt.xml')
detection = face_detector.detectMultiScale(image, scaleFactor=1.1 )

for x,y,w,h in detection:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,200), 2)
cv2.imshow('', image)
cv2.waitKey(0)