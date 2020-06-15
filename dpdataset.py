import cv2
import os

cap=cv2.VideoCapture(0)
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id=1

count=0
while(True):
	ret,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_classifier.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		count+=1
		#saving image along with co-ordinates
		cv2.imwrite("data/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w])
		cv2.imshow('image',img)
	k=cv2.waitKey(100) & 0xFF
	if k==27:
		break


cap.release()
cv2.destroyAllWindows()
