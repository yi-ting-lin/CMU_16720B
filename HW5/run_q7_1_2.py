import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets

#reference:https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


def cal_loss_acc(x,y,model):
    #Get y_hat and target
    model.eval()
    y_hat = model(x)

    #calcuate loss and acc
    loss = nn.functional.cross_entropy(y_hat, y)
    predict_class = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_class == y).item()

    return loss, correct_cnt

def train_step(x,y,model,optimizer):
    #Set modl to train mode
    model.train()
            
    #Get y_hat and target
    y_hat = model(x)

    #calcuate loss and acc
    loss = nn.functional.cross_entropy(y_hat, y)
    predict_class = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_class == y).item()

    # backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), correct_cnt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.fc1 = nn.Sequential(nn.Linear(1568, 10))
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 1568)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    # settings
    max_iters = 20
    batch_size = 32
    learning_rate = 0.01

    #Data loader
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    #model & optimizaer
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    

    #Training
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
           
        acc = acc/len(train_set)
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
    for x,y in test_data_loader:
        _, acc = cal_loss_acc(x,y, model)
        test_acc += acc
    
    test_acc /= len(test_set)

    print('Test accuracy: ' + str(test_acc))




