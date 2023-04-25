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
batch_size = 1
path = '.\\model_mask.pth'
path_data = '.\\data_mask\\test\\'

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
}
dataset = torchvision.datasets.ImageFolder(path_data, transform=data_transforms['test'])

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

net = torchvision.models.resnet50()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
model = net
model.load_state_dict(torch.load(path), strict=False)
model.eval()


criterior = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0., momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

acc = []
loss = []

def test(model, test_loader):
    model = model.to(device)
    correct = 0.0
    total = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_loss = criterior(outputs, labels).detach().cpu().numpy()
        _, test_predictions = torch.max(outputs.data, 1)
        # test_predictions = torch.argmax(outputs, dim=1)
        test_accuracy = torch.mean((test_predictions == labels).float()).detach().cpu().numpy()
        acc.append(test_accuracy)
        loss.append(test_loss)
        total += labels.size(0)
        correct += (test_predictions == labels).sum().item()
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        # print(f"Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}")

import matplotlib.pyplot as plt
def showplt():

    plt.title("test")

    plt.plot(acc,label="acc",color = 'r')
    plt.plot(loss,label="loss", color = 'g')

    plt.xlabel("images")
    plt.ylabel("values")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    test(model, test_loader)
    showplt()



