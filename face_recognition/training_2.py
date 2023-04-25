from torch import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import os
import glob

path_face = '.\\model.pt'  # mô hình hiện tại training có 3 đầu ra

# transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
}
# lấy tên file được tạo mới nhất
list_of_files = glob.glob('.\\data\\addface\\*')
latest_file = max(list_of_files, key=os.path.getctime)
path = str(latest_file)
name = path[15:]
# file image
path_data = '.\\data\\addface'
# path file training xong sẽ xoá file ảnh đi
path_rm = os.path.join(path_data+'\\'+ name)

# tạo dataloader
dataset = torchvision.datasets.ImageFolder(path_data, transform=data_transforms['train'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2)

# tính số đầu ra, bằng tổng class của train và add face
# add face là folder để chứa những ảnh được thêm sau

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#loss
criterior = nn.CrossEntropyLoss()
#model
net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 3) 

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
model = net
checkpoint = torch.load(path_face)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train() #  tiếp tục training

def train(model,dataloader, criterion, num_epochs):
    running_loss = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
    }, path_face)
    print('successfully saved')
    os.remove(path_rm)

if __name__ == '__main__':
    train(model,dataloader, criterior,num_epochs=10 )


