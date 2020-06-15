import cv2
import numpy as np
from PIL import Image
import os

path='data'
clf=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def get_image_and_label(path):
	imagepath=[os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	ids=[]
	for image in imagepath:
		PIL_img=Image.open(image).convert('L')#here L is grayscale
		img_numpy=np.array(PIL_img,'uint8')
		id=int(os.path.split(image)[-1].split(".")[1])
		face=detector.detectMultiScale(img_numpy)
		for (x,y,w,h) in face:
			faces.append(img_numpy[y:y+h,x:x+w])
			ids.append(id)

	return faces,ids

print("\n Training in process...")

face,ids=get_image_and_label(path)
#ids=[0]*len(face)
clf.train(face,np.array(ids))

clf.write('trainer.yml')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


