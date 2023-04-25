import datetime
import time
import numpy as np
from torch import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import cv2
import PIL
from cvzone.FaceDetectionModule import FaceDetector
from keras.preprocessing import image

path = '.\model_mask.pth'
batch_size = 1

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()

])

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model = net
model.load_state_dict(torch.load(path), strict=False)
model.eval()



# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

class_id = ['with_mask', 'without_mask']

def argmax(model,inputs):
    output = model(inputs)
    max_id = np.argmax(output.detach().numpy())
    # output_arr = output.detach().numpy()
    # out_sorted = np.sort(output_arr)
    #
    # max1 = out_sorted[0][-1]
    # max2 = out_sorted[0][-2]
    # result = max1 - max2

    predicted_label = class_id[max_id]

    return predicted_label


def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

cap = cv2.VideoCapture(0)  # Set the webcam
Webcam_720p()
fps = 0
sequence = 0
result = ''
score = 0.0


cap =cv2.VideoCapture(0)
face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _ ,img =cap.read()
    face =face_cascade.detectMultiScale(img ,scaleFactor=1.1 ,minNeighbors=4)
    for(x ,y ,w ,h) in face:
        face_img = img[y: y +h, x: x +w]
        cv2.imwrite('temp.jpg' ,face_img)
        test_image =image.load_img('temp.jpg' ,target_size=(150 ,150 ,3))
        test_image =image.img_to_array(test_image)
        test_image =np.expand_dims(test_image ,axis=0)
        pred =mymodel.predict(test_image)[0][0]
        if pred==1:
            cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(0 ,0 ,255) ,3)
            cv2.putText(img ,'NO MASK' ,(( x +w )//2 , y + h +20) ,cv2.FONT_HERSHEY_SIMPLEX ,1 ,(0 ,0 ,255) ,3)
        else:
            cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(0 ,255 ,0) ,3)
            cv2.putText(img ,'MASK' ,(( x +w )//2 , y + h +20) ,cv2.FONT_HERSHEY_SIMPLEX ,1 ,(0 ,255 ,0) ,3)
        datet =str(datetime.datetime.now())
        cv2.putText(img ,datet ,(400 ,450) ,cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,255 ,255) ,1)

    cv2.imshow('img' ,img)

    if cv2.waitKey(1 )==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()