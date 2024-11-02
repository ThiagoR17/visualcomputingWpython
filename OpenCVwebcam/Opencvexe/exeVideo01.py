import cv2

webcam = cv2.VideoCapture(0)

while True:
    cam, frame = webcam.read()

    cv2.imshow("camTHI", frame)

    if cv2.waitKey(1) == ord('f'):
        break

webcam.release()
cv2.destroyAllWindows()