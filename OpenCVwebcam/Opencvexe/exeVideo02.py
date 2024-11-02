import cv2

webcam = cv2.VideoCapture(0)

facevideo = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

while True:
    cam, frame  = webcam.read()

    cinza = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detecta = facevideo.detectMultiScale(cinza)
    for(x,y,l,a) in detecta:
        cv2.rectangle(frame,(x,y), (x+l, y+a), (255, 0, 0), 2)

    cv2.imshow("camTHI", frame)
    if cv2.waitKey(1) & 0xFF == ord("f"):
        break
webcam.release()
cv2.destroyAllWindows()