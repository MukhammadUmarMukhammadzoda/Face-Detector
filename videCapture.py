import cv2 as cv

face_detector = cv.CascadeClassifier('cascade/data/haarcascade_frontalface_alt.xml')
video_capture = cv.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(image)
    for (x,y,h,w) in detections:
        cv.rectangle(frame, (x,y), (x+h, y+w), (0,120,200), 2)

        cv.imshow('video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break