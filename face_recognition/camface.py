import numpy as np
import torch
import torch.nn
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import PIL
import cv2
from facedetaction import FaceDetector
import telebot
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time
import os

bot = telebot.TeleBot('5519458842:AAF6VgxTkUptS9VmzuM17abqCq6uRbrgiS8')
chat_id = 1942368034

cred = credentials.Certificate('iotfirebase.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-kmt-default-rtdb.firebaseio.com/key/-NPVMhvME0K94HRLCIJ5'
})
ref = db.reference('key')
box_ref = ref.child('key')
# Let's preprocess the inputted frame
data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor()
])

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
path = '.\\model_mask.pth'
model = net
model.load_state_dict(torch.load(path))
model.eval()

list_train = os.listdir('.\\data\\train')
print(list_train)
number_train = len(list_train)
out = int(number_train)
net_face = torchvision.models.resnet50(pretrained=True)
num = net_face.fc.in_features
net_face.fc = nn.Linear(num, out)
path = '.\\model_face1.pth'
model_face = net_face
model_face.load_state_dict(torch.load(path))
model_face.eval()

# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

class_id = ['mask','nomask']

def argmax(model,inputs):
    output = model(inputs)
    max_id = np.argmax(output.detach().numpy())
    output_arr = output.detach().numpy()
    out_sorted = np.sort(output_arr)

    max1 = out_sorted[0][-1]
    max2 = out_sorted[0][-2]
    result = max1 - max2
    predicted_label = class_id[max_id]
    return result,predicted_label
names = list_train
def argmax_face(model,inputs):
    output = model(inputs)
    max_id = np.argmax(output.detach().numpy())
    output_arr = output.detach().numpy()
    out_sorted = np.sort(output_arr)

    max1 = out_sorted[0][-1]
    max2 = out_sorted[0][-2]
    result = max1 - max2
    if result >0.5:
        predicted_label = names[max_id]
    else:
        predicted_label = 'unknown'
    return result,predicted_label

def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor
cap = cv2.VideoCapture(0)  # Set the webcam
Webcam_720p()
result = ''
score = 0.0
detector = FaceDetector()
while True:
    ret, frame = cap.read()  # Capture each frame
    img, bboxs = detector.findFaces(frame)
    if bboxs:
        x, y, w, h = bboxs[0]['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        image = frame[y: y + h, x: x + w]
        image_data = preprocess(image)
        score, result = argmax(model,image_data)
        if result == 'nomask':
            score_face, name = argmax_face(model_face, image_data)
            cv2.putText(frame, '%s' % (name), (950, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv2.putText(frame, '(score = %.5f)' % (score), (950, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # if name != 'unknown':
            #     now = datetime.datetime.now()
            #     txt = f'{name} đã mở cửa lúc {now} '
            #
            #     bot.send_message(chat_id, txt)
            #     box_ref.update({
            #         'stt': 1
            #     })
            #     time.sleep(15)
            #     box_ref.update({
            #         'stt': 0
            #     })

        if result == 'mask':
            print('Bỏ khẩu trang ra')
    cv2.putText(frame, '%s' % (result), (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(frame, '(score = %.5f)' % (score), (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
    cv2.imshow("DETECTER", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow()