import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#reference:https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


def cal_loss_acc(x,y,model):
    #set model to evaluate mode
    model.eval()
    #Get y_hat and target
    y_hat = model(x)
    gold_labels = torch.max(y, 1)[1]

    #calcuate loss and acc
    loss = nn.functional.cross_entropy(y_hat, gold_labels)    
    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels==gold_labels).item()

    return loss, correct_cnt

def train_step(x,y,model,optimizer):
    #Set modl to train mode
    model.train()
            
    #Get y_hat and target
    y_hat = model(x)
    gold_labels = torch.max(y, 1)[1]

    #calcuate loss and acc
    loss = nn.functional.cross_entropy(y_hat, gold_labels)    
    predict_labels = torch.max(y_hat, 1)[1]
    correct_cnt = torch.sum(predict_labels==gold_labels).item()

    # backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), correct_cnt


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    #Data parsing
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    test_data = scipy.io.loadmat('../data/nist36_test.mat')
    train_x = train_data['train_data']
    train_y = train_data['train_labels']
    test_x  = test_data['test_data']
    test_y  = test_data['test_labels']

    
    # settings
    max_iters = 50
    batch_size = 32
    learning_rate = 0.01
    hidden_size = 64

    #Data loader
    train_x = torch.from_numpy(train_x).type(torch.float32)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    train_data_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,num_workers=2)

    test_x = torch.from_numpy(test_x).type(torch.float32)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)
    test_data_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True, num_workers=2)

    #model & optimizaer
    model = Model(input_size=train_x.shape[1], hidden_layer_size=hidden_size, output_size=train_y.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    num_train_data = train_y.shape[0]

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
    
    test_acc /= test_y.shape[0]

    print('Test accuracy: ' + str(test_acc))




