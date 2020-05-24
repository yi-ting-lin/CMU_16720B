import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np

#reference:https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


def cal_loss_acc(x,y,model):
    #Get y_hat and target
    model.eval()
    y_hat = model(x)

    #Get y_hat
    y_hat = model(x)
    #calcuate loss and acc
    gold_labels = torch.max(y, 1)[1]
    loss = nn.functional.cross_entropy(y_hat, gold_labels)
    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels == gold_labels).item()

    return loss, correct_cnt

def train_step(x,y,model,optimizer):
    #Set modl to train mode
    model.train()
            
    #Get y_hat
    y_hat = model(x)
    #calcuate loss and acc
    gold_labels = torch.max(y, 1)[1]
    loss = nn.functional.cross_entropy(y_hat, gold_labels)
    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels == gold_labels).item()



    # backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), correct_cnt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.fc1 = nn.Sequential(nn.Linear(4096, 1024))
        self.fc2 = nn.Sequential(nn.Linear(1024, 36))


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # settings
    max_iters = 50
    batch_size = 32
    learning_rate = 0.01

    #Load data
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')

    train_x = train_data['train_data']
    train_y = train_data['train_labels']
    test_x  = test_data['test_data']
    test_y  = test_data['test_labels']

    #reshape feautres into 32x32
    num_train_data = train_x.shape[0]
    train_x_reshape = []
    for i in range(num_train_data):
        train_x_reshape.append(train_x[i].reshape(32,32))
    train_x_reshape = np.array(train_x_reshape)

    num_test_data = test_x.shape[0]
    test_x_reshape = []
    for i in range(num_test_data):
       test_x_reshape.append(test_x[i].reshape(32,32))
    test_x_reshape = np.array(test_x_reshape)


    #Data loader
    train_x_reshape = torch.from_numpy(train_x_reshape).type(torch.float32).unsqueeze(1)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    train_data_loader = DataLoader(TensorDataset(train_x_reshape, train_y), batch_size=batch_size, shuffle=True,num_workers=2)

    test_x_reshape = torch.from_numpy(test_x_reshape).type(torch.float32).unsqueeze(1)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)
    test_data_loader = DataLoader(TensorDataset(test_x_reshape, test_y), batch_size=batch_size, shuffle=True, num_workers=2)

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
    for x,y in test_data_loader:
        _, acc = cal_loss_acc(x,y, model)
        test_acc += acc
    
    test_acc /= num_test_data

    print('Test accuracy: ' + str(test_acc))

    #Save parameters
    torch.save(model.state_dict(), "q7_1_3_model_parameter.pkl")




