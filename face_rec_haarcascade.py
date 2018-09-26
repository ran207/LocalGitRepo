import cv2

proj_path='/home/rachna/PycharmProjects/cv2_testing/'
casc_path=proj_path+'cascade/haarcascade_frontalface_default.xml'

# create face cascade object
face_cascade=cv2.CascadeClassifier(casc_path)
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

#create object for webcam capture
webcam_stream=cv2.VideoCapture(0)

while True:
    # begin capture
    _,frame=webcam_stream.read()

    # convert bgr to grayscale
    gray_ver=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # call method to detect faces and create list of rectangle measurements
    face_rec=face_cascade.detectMultiScale(
        gray_ver,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(50,50),
        flags=cv2.CASCADE_SCALE_IMAGE #check this line later
    )

    # make the rectangles in the img
    for (x, y, w, h) in face_rec:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 200), 3)
        eye_rec=eye_cascade.detectMultiScale(gray_ver[y:y+h,x:x+w])
        for (ex,ey,ew,eh) in eye_rec:
            cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0, 0, 200), 3)
        # shows only the face area
        # cv2.imshow('Webcam feed',frame[y:y+w, x:x+h])

    # display the webcam feed frame by frame
    cv2.imshow('Webcam feed',frame)

    # stop while loop when q is hit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_stream.release()
cv2.destroyAllWindows()
