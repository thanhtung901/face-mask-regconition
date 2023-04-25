import random
from flask import Flask, render_template, Response, request, flash
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import torch
import torch.nn
import torchvision
from torch import nn, optim
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

random.seed(123)
global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
id = random.randint(0, 10000000)
id2 = random.randint(0, 20000000)

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

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

netmask = torchvision.models.resnet50(pretrained=True)
num_ftrs = netmask.fc.in_features
netmask.fc = nn.Linear(num_ftrs, 2)
path = '.\\model_mask.pth'
model = netmask
model.load_state_dict(torch.load(path))
model.eval()


# Set the Webcam

def train():
    num_epochs = 5
    path_data = '.\\data\\train'
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(225),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    }
    dataset = torchvision.datasets.ImageFolder(path_data, transform=data_transforms['train'])

    val_split = 0.1
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_set, val_set = torch.utils.data.random_split(dataset, [len(train_indices), len(val_indices)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)

    dataloader_dict = {"train": train_loader, "val": validation_loader}

    list_train = os.listdir('.\\data\\train')
    number_train = len(list_train)
    out = int(number_train)
    print('Number', out)
    if out > 6:
        print('Số người đã đạt giới hạn')
    else:
        net = torchvision.models.resnet50(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out)

        criterior = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=net.parameters(), lr=0.001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        best_acc = 0.0
        model = net.to(device)
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in dataloader_dict[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    la = labels.detach().cpu()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterior(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).detach().cpu()

                epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloader_dict[phase].dataset)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
        torch.save(model.state_dict(), '.\\model_face1.pth')

def Webcam_720p():
    camera.set(3, 1280)
    camera.set(4, 720)

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

def argmax_face(model,inputs):
    list_train = os.listdir('.\\data\\train')
    names = list_train
    output = model(inputs)
    max_id = np.argmax(output.detach().numpy())
    output_arr = output.detach().numpy()
    out_sorted = np.sort(output_arr)

    max1 = out_sorted[0][-1]
    max2 = out_sorted[0][-2]
    result = max1 - max2
    if result >0.4:
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


Webcam_720p()
result = ''
score = 0.0
detector = FaceDetector()


def add_face(frame):
    folder = f'.\\data\\train\\{str(id2)}'
    os.mkdir(folder)
    time.sleep(3)
    count = 0
    while (True):
        img, bboxs = detector.findFaces(frame)
        if bboxs:
            x, y, w, h = bboxs[0]['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f".\\data\\train\\{str(id2)}\\user." + str(id) + '.' + str(count) + ".jpg",
                        img[y:y + h, x:x + w])
        if count >= 20:
            time.sleep(3)
            print("successfully")
            bot.send_message(chat_id, f'đã thêm khuân mặt thành công id:  {id2}')
            break
    return frame

list_train = os.listdir('.\\data\\train')
number_train = len(list_train)
out = int(number_train)

net_face = torchvision.models.resnet50(pretrained=True)
num = net_face.fc.in_features
net_face.fc = nn.Linear(num, out)
path = '.\\model_face1.pth'
model_face = net_face
model_face.load_state_dict(torch.load(path))
model_face.eval()

def detect_face(frame):
    while True:
        img, bboxs = detector.findFaces(frame)
        if bboxs:
            x, y, w, h = bboxs[0]['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            image = frame[y: y + h, x: x + w]
            image_data = preprocess(image)
            score, result = argmax(model, image_data)
            if result == 'nomask':
                score_face, name = argmax_face(model_face, image_data)
                print(name)
                cv2.putText(frame, '%s' % (name), (950, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                cv2.putText(frame, '(score = %.5f)' % (score), (950, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if name != 'unknown':
                    now = datetime.datetime.now()
                    txt = f'{name} đã mở cửa lúc {now} '

                    bot.send_message(chat_id, txt)
                    box_ref.update({
                        'stt': 1
                    })
                    time.sleep(30)
                    box_ref.update({
                        'stt': 0
                    })

            if result == 'mask':
                print('Bỏ khẩu trang ra')
        cv2.putText(frame, '%s' % (result), (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, '(score = %.5f)' % (score), (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
        # cv2.imshow("DETECTER", frame)
        return frame

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec

    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (rec):
                frame = add_face(frame)
            if (capture):
                train()
                bot.send_message(chat_id, f'training oke')
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('face') == 'Predict':
            global face
            face = not face
            if (face):
                time.sleep(10)
        elif request.form.get('rec') == 'Add':
            global rec
            rec = not rec
            if (rec):
                time.sleep(10)
        elif request.form.get('train') == 'train':
            global capture
            capture = not capture

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()