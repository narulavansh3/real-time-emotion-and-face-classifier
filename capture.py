
import cv2
import numpy as np
cam=cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



id=input('enter user id')
sampleNum=0

while(True):
	ret,img=cam.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		sampleNum=sampleNum+1
		cv2.imwrite("dataset/user."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.waitKey(100)
	cv2.imshow("face",img)
	cv2.waitKey(1)
	if(sampleNum>500):
		break

cam.release()
cv2.destroyAllWindows()
