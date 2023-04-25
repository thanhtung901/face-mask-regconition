
import time
import numpy as np
from torch import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms

resize = 224
num_epochs = 10

path_save = '.\\model_mask.pth'
path_data = '.\\data_mask\\train'
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=False)

dataloader_dict = {"train": train_loader, "val": validation_loader}

net = torchvision.models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)


criterior = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

losses = {"train":[], "val":[]}
acces = {"train":[], "val":[]}
import matplotlib.pyplot as plt

def train(model, dataloader_dict, criterion, optimizer, num_epochs):
    begin = time.time()
    best_acc =0.0
    best_val = 0.0
    model = model.to(device)
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

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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
            losses[phase].append(epoch_loss)
            acces[phase].append(epoch_acc)
            if phase == 'val' and epoch_acc > best_val:
                best_val = epoch_acc
            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val))
    print('Best train  Acc: {:4f}'.format(best_acc))
    torch.save(model.state_dict(), path_save)

def showplt():

    plt.title("Training and Validation mask")

    plt.plot(acces["train"],label="train_acc",color = 'r')
    plt.plot(losses["train"],label="train_loss", color = 'g')

    plt.plot(acces["val"],label="val_acc", color = 'b')
    plt.plot(losses["val"],label="val_loss", color = 'm')

    plt.xlabel("epoch")
    plt.ylabel("values")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train(net,dataloader_dict, criterior, optimizer, num_epochs)
    showplt()
