import cv2
import os
import time
import random
from facedetaction import FaceDetector
# from cvzone.FaceDetectionModule import FaceDetector
id = random.randint(0, 10000000)
id2 = random.randint(0, 20000000)
def names ():
    name = input('Name : ' )
    return name
namess = names()
folder = f'.\\data\\train\\{str(namess)}'
os.mkdir(folder)

def add_face(name):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    detector = FaceDetector()
    count = 0
    while(True):
        ret, img = cam.read()
        img, bboxs = detector.findFaces(img)
        if bboxs:
            x, y, w, h = bboxs[0]['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f".\\data\\train\\{str(name)}\\user." + str(id) + str(id2) +'.' + str(count) + ".jpg", img[y:y+h,x:x+w])
            print("adđ oke")
            cv2.imshow('image', img)

        if count >= 10:
            print("successfully")
            break
    cam.release()

add_face(namess)
cv2.waitKey(0)
cv2.destroyAllWindows()


