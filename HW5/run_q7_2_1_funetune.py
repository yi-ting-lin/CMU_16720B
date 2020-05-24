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
        golden_label = torch.autograd.Variable(y.type(torch.FloatTensor).long())

        #calcuate loss and acc
        y_hat = model(x_var)
        predict_label = torch.max(y_hat, 1)[1]
        acc += torch.sum(predict_label == golden_label).item()
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


if __name__ == '__main__':
    # settings
    num_epoch1 = 10
    num_epoch2 = 10
    batch_size = 32
    num_workers = 4
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
    model = torchvision.models.squeezenet1_1(pretrained=True)
    num_classes = len(train_set.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes


    #set pre-trained param fixed
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    print('=========Training on the classifier only with lr = 0.001=========')
    train_losses = []
    train_accs = []
    for itr in range(num_epoch1):
        total_loss = run_epoch(train_loader, model, optimizer)
        acc = cal_acc(train_loader, model)
        train_losses.append(total_loss)
        train_accs.append(acc)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

    #set pre-trained param trainable
    for param in model.parameters():
        param.requires_grad = True

    print('=========Training on both the feautres and the classifier with lr = 1e-5========')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for itr in range(num_epoch1, num_epoch1+num_epoch2):
        total_loss = run_epoch(train_loader, model, optimizer)
        acc = cal_acc(train_loader, model)
        train_losses.append(total_loss)
        train_accs.append(acc)
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

    #Save parameters
    torch.save(model.state_dict(), "q7_2_finetune_model_parameter.pkl")

    plt.figure('accuracy')
    plt.plot(range(num_epoch1+num_epoch2), train_accs, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Averge accuracy')
    plt.show()

    plt.figure('loss')
    plt.plot(range(num_epoch1+num_epoch2), train_losses, color='r')
    plt.xlabel('# of epoch')
    plt.ylabel('Cross entropy loss')
    plt.show()

    valid_acc = cal_acc(valid_loader, model)
    print('Validation accuracy: ' + str(valid_acc))

    test_acc = cal_acc(test_loader, model)
    print('Test accuracy: ' + str(test_acc))
    




