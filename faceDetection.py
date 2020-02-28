import numpy as np
import cv2
from tensorflow import keras
faceCascade=cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
"""imgPath = "WholeImages/images (6).jpeg"
img = cv2.imread(imgPath)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.5,5)
for (x,y,w,h) in faces:
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	face = img[y:(y+h) , x:(x+w)]
	cv2.imshow("image2",face)
	cv2.imwrite("Testing Images/16.jpg",face)

cv2.imshow("image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()  
"""
modelPath = "Models/TSModel2"
model = keras.models.load_model(modelPath)
cap = cv2.VideoCapture(0)

def preprocessingImageToClassifier(image=None,imageSize=48,mu=92.31604903840567,std=70.85156431910688):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image


mu = 130.2243228911731
std = 74.46392763687916
while True:
    _,frame = cap.read()
    if _ == True:
	    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	    faces = faceCascade.detectMultiScale(gray,1.5,5)
	    for (x, y, w, h) in faces:
	    	face = frame[y:(y+h) , x:(x+w)]
	    	face = preprocessingImageToClassifier(image = face,
                                    imageSize = 48,
                                    mu = mu,
                                    std = std)
	    	pred = model.predict(face) * 100
	    	print("Angry = "+str(pred[0][0])+"%")
	    	print("Happy = "+str(pred[0][1])+"%")
	    	print("Sad = "+str(pred[0][2])+"%")
	    	print("Surprice = "+str(pred[0][3])+"%")
	    	print("-----------------------------")
	    	frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	    cv2.imshow("video",frame)

	    if cv2.waitKey(20) & 0xFF ==ord('q'):
	        break

    else:
	    break

cap.release()
cv2.destroyAllWindows()  
