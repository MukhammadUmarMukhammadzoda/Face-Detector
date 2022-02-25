import cv2

#Creating Variables
img = 'Images/img2.jpg'
image = cv2.imread(img)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt.xml')
face_detection = face_detector.detectMultiScale(image, scaleFactor=1.04, minNeighbors=2, maxSize=(60,60))
eye_detector = cv2.CascadeClassifier('cascade/data/haarcascade_eye.xml')
eye_detection = eye_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3, maxSize=(30,30))
video_capture = cv2.VideoCapture(0)


#Drawing Rectangle for Faces
for x,y,w,h in face_detection:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,200), 2)


#Drawing Rectangle for eyes

for x,y,w,h in eye_detection:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 1)


cv2.imshow('', image)
cv2.waitKey(0)
