
import dlib
import cv2
# from matplotlib import pyplot as plt

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# construct the argument parser and parse the arguments
video=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()

# detect faces in the grayscale image

# loop over the face detections
while True:
    ret,image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for  rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        print( x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Vansh"
        elif(id==2):
            id="Manish"
        elif(id==3):
            id="Prakhar"

        cv2.putText(image,str(id),(x,y+h),font,4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("face",image)
    if(cv2.waitKey(1)==ord('q')):
        break   
cam.release()
cv2.destroyAllWindows()
