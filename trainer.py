
import os #to communicate with operating system
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create() #local binary pattern histogram
path='dataset'
def getImagesWithID(path):
	imagepaths=[os.path.join(path,f) for f in os.listdir(path)] #it joins all the data in one file
	faces=[]
	IDs=[]
	for imagePath in imagepaths:
		faceimg=Image.open(imagePath).convert('L') #l=grayscale
		faceNp=np.array(faceimg,'uint8')       #cv2 is not compatible with normal array ie why we use numpy array
		ID=int(os.path.split(imagePath)[-1].split('.')[1]) #descending order of images ie starting from last to get id with index =[1]
		faces.append(faceNp)
		print (ID)
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return IDs,faces
IDs,faces= getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainingData.yml')  #yml file is made
cv2.destroyAllWindows()

