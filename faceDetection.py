import numpy as np
import cv2
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

cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    if _ == True:
	    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	    faces=faceCascade.detectMultiScale(gray,1.5,5)
	    for (x,y,w,h) in faces:
		    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	    cv2.imshow("video",frame)

	    if cv2.waitKey(20) & 0xFF ==ord('q'):
	        break

    else:
	    break

cap.release()
cv2.destroyAllWindows()  
