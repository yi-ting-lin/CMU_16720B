import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
from torchvision.datasets import ImageFolder

#reference:https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c


def cal_acc(loader, model):
    model.eval()
    num_samples = 0
    acc = 0

    for x,y in loader:
        x_var = torch.autograd.Variable(x.type(torch.FloatTensor), volatile=True)
        golden_labels = torch.autograd.Variable(y.type(torch.FloatTensor).long())
        
        #calcuate loss and acc
        y_hat = model(x_var)
        predict_labels = torch.max(y_hat, 1)[1]
        acc += torch.sum(predict_labels == golden_labels).item()
        num_samples += x.size(0)

    acc /= num_samples
    return acc

def run_epoch(loader,model,optimizer):
    #Set modl to train mode
    model.train()
            
    total_loss = 0
    
    for x,y in loader:
        x = torch.autograd.Variable(x.type(torch.FloatTensor))
        y = torch.autograd.Variable(y.type(torch.FloatTensor).long())

        #get y_hat and calculate loss
        y_hat = model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        total_loss += loss.item()

        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(3,6,5,1,2)
        self.conv2=nn.Conv2d(6,16,5,1,2)
        self.conv3=nn.Conv2d(16,120,5,1,2)
        self.fc1=nn.Linear(94080,84)
        self.fc2=nn.Linear(84, 17)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        x=x.view(-1, 94080)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

if __name__ == '__main__':
    # settings
    num_epoch = 100
    batch_size = 64
    learning_rate = 0.01
    num_workers = 4
    learning_rate = 0.001
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


    #Directories
    train_dir = '../data/oxford-flowers17/train'
    valid_dir = '../data/oxford-flowers17/val'
    test_dir = '../data/oxford-flowers17/test'
    #Data loader
    train_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    train_set = ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True)

    valid_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    valid_set = ImageFolder(valid_dir, transform=valid_transform)
    valid_loader = DataLoader(valid_set,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)
    
    test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    test_set = ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_set,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=False)
    #model & optimizaer
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accs = []
    for itr in range(num_epoch):
        total_loss = run_epoch(train_loader, model, optimizer)
        acc = cal_acc(train_loader, model)
        train_losses.append(total_loss)
        train_accs.append(acc)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

    #Save parameters
    torch.save(model.state_dict(), "q7_2_scratch_parameter.pkl")

    plt.figure('accuracy')
    plt.plot(range(num_epoch), train_accs, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Averge accuracy')
    plt.show()

    plt.figure('loss')
    plt.plot(range(num_epoch), train_losses, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Cross entropy loss')
    plt.show()

    valid_acc = cal_acc(valid_loader, model)
    print('Validation accuracy: ' + str(valid_acc))

    test_acc = cal_acc(test_loader, model)
    print('Test accuracy: ' + str(test_acc))
    




