import cv2

webcam = cv2.VideoCapture(0)
videoface = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
scamEye = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

while True:
    cam, frame = webcam.read()
    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scam = videoface.detectMultiScale(gray)
    for(x,y,l,a) in scam:
        cv2.rectangle(frame,(x,y),(x+l,y+a),(0,255,0),2)
        pegaOlho = frame[y:y+l,x:x+l]
        Olho = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
        localizaOlho = scamEye.detectMultiScale(Olho)
        for(ox, oy, ol, oa) in localizaOlho:
            cv2.rectangle(pegaOlho, (ox, oy), (ox+ol,oy+oa), (255,0,0),2)
    cv2.imshow("webmTHI", frame)
    if cv2.waitKey(1) == ord('f'):
        break

webcam.release()
cv2.destroyAllWindows()