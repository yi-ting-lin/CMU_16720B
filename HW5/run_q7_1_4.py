import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import string

import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches

from q4 import *

#reference:https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


def cal_loss_acc(x,y,model):
    #Get y_hat and target
    model.eval()
    y_hat = model(x)

    #calcuate loss and acc
    loss = nn.functional.cross_entropy(y_hat, y)
    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels == y).item()

    return loss, correct_cnt

def train_step(x,y,model,optimizer):
    #Set modl to train mode
    model.train()
            
    # get output
    y_hat = model(x)
    loss = nn.functional.cross_entropy(y_hat, y)

    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels==y).item()

    # backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), correct_cnt


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,47)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1, 256)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

def get_letters_from_image(im_path):
    im1 = skimage.img_as_float(skimage.io.imread(im_path))
    bboxes, bw = findLetters(im1)
    
    plt.imshow(bw, cmap='gray')
    
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    
    # find the rows using..RANSAC, counting, clustering, etc.
    heights = [bbox[2] - bbox[0] for bbox in bboxes]
    min_height = min(heights)

    # (centerX, centerY, width, height)
    pos_list = [((bbox[3]+bbox[1])//2, (bbox[2]+bbox[0])//2, bbox[3]-bbox[1], bbox[2]-bbox[0]) for bbox in bboxes]

    # sort by centerY
    pos_list = sorted(pos_list, key=lambda x: x[1])
    rows = []
    row = []
    pre_y = None
    for pos in pos_list:
        if pre_y == None or pos[1] - pre_y < min_height: # still in the same row
            row.append(pos)
        else: #changing to next row
            #sorted by centerX and push into rows
            row = sorted(row,key=lambda x: x[0])
            rows.append(row)
            row = [pos]
        #update pre_y
        pre_y = pos[1]

    #sort andpush the last row
    row = sorted(row,key=lambda x: x[0])
    rows.append(row)


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    raw_data = []
    for row in rows:
        row_raw_data = []
        for x, y, w, h in row:
            crop_roi = bw[y-h//2:y+h//2, x-w//2:x+w//2]
            
            #padding
            if h < w:
                w_pad = w//16
                h_pad = (w-h)//2+w_pad
            else:
                h_pad = h//16
                w_pad = (h-w)//2+h_pad
            crop_roi = np.pad(crop_roi, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            # resize to 32*32
            crop_roi = skimage.transform.resize(crop_roi, (28, 28))
            crop_roi = skimage.morphology.erosion(crop_roi, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            crop_roi = crop_roi.T
            crop_roi = 1.0 - crop_roi
            row_raw_data.append(crop_roi)
        raw_data.append(np.array(row_raw_data))

    return raw_data

def letter_recognition(input_data):

    letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in string.ascii_lowercase[:26]])

    inputs = []
    for letter_data in input_data:
        #expand it dim to 28x28x1
        letter_data = np.expand_dims(letter_data, axis=2)

        #Transform data type
        input_letter = transform(letter_data).type(torch.float32)
        inputs.append(input_letter)
    inputs = torch.stack(inputs, dim=0)

    y_hats = model(inputs)
    predict_labels = torch.max(y_hats, 1)[1]
    predict_labels = predict_labels.numpy()
    row_s = ''
    for i in range(predict_labels.shape[0]):
        row_s += letters[int(predict_labels[i])]

    print(row_s)


if __name__ == '__main__':
    # settings
    max_iters = 50
    batch_size = 64
    learning_rate = 0.01

    #Data loader
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True,
                                      download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

    test_set = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False,
                                     download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

    #model & optimizaer
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    

    #Training
    num_train_data = len(train_set)
    train_losses = []
    train_accs = []
    for itr in range(max_iters):
        total_loss = 0
        acc = 0
        for X, Y in train_data_loader:
            #Train step
            loss, correct_cnt = train_step(X,Y,model,optimizer)
            total_loss += loss
            acc += correct_cnt
           
        acc = acc/num_train_data
        train_losses.append(total_loss)
        train_accs.append(acc)

        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))


    plt.figure('accuracy')
    plt.plot(range(max_iters), train_accs, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Averge accuracy')
    plt.show()

    plt.figure('loss')
    plt.plot(range(max_iters), train_losses, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Cross entropy loss')
    plt.show()

    test_acc = 0
    num_test_data = len(test_set)
    for x,y in test_data_loader:
        _, acc = cal_loss_acc(x,y, model)
        test_acc += acc

    test_acc /= num_test_data

    print('Test accuracy: ' + str(test_acc))


    #Evaluate on the given images
    for img in os.listdir('../images'):
        img_path = os.path.join('../images',img)
        input_data = get_letters_from_image(img_path)

        for row_data in input_data:
            #print(row_data.shape)
            letter_recognition(row_data)


    #Save parameters
    torch.save(model.state_dict(), "q7_1_4_model_parameter.pkl")

