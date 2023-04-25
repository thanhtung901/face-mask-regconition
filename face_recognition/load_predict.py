import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import models, transforms

class_id = ['tung','trang','tuyen']
class Predict():
    def __init__(self, class_id):
        self.class_id = class_id

    def predict_labels(self, output):
        max_id = np.argmax(output.detach().numpy())
        output_arr = output.detach().numpy()
        out_sorted = np.sort(output_arr)

        max1 = out_sorted[0][-1]
        max2 = out_sorted[0][-2]
        result = max1-max2
        print('result', result)
        if result >0.5:
            predicted_label = self.class_id[max_id]
        else:
            predicted_label = 'unknown'
        return predicted_label
predictor = Predict(class_id)
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor()])
    return my_transforms(image_bytes).unsqueeze(0)
path = '.\\model_face.pth'

def predict(img):
    net = torchvision.models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)  # output 4 class

    model = net
    model.load_state_dict(torch.load(path))
    model.eval()

    data_input = transform_image(img)

    output = model(data_input)
    label = predictor.predict_labels(output)
    return label

if __name__ == '__main__':
    img = Image.open(r".jpg")
    # img.show()
    name = predict(img)
    print(name)
