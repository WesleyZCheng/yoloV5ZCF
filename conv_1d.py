#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
import pandas as pd

torch.manual_seed(1) #reproducible

#Hyper Parameters
EPOCH = 10
BATCH_SIZE = 150
BATCH_SIZE_te = 100
LR = 0.001

expName = './cnn1-3_nlnn_gaussian_embedding_Relu_new'
if expName.find('gaussian_embedding') != -1:
    model_path = './output/cnn/train_epoch_0_step_0.pth'

if not os.path.exists('./output/{}'.format(expName)):
    os.mkdir('./output/{}'.format(expName))

class SBPEstimateDataset(Dataset):
    def __init__(self):
        # data = sio.loadmat('data.mat')
        # self.train_x = data['DS_Train']
        # self.train_y = data['yTr']
        trainPath = 'D:\\ZHY\\train.xlsx'
        self.train = pd.read_excel(trainPath).values


    def __len__(self):
        return len(self.train[:,-1])

    def __getitem__(self):
        train = self.train
        """Convert ndarrays to Tensors."""
        return torch.from_numpy(train_x).float(),

class SBPEDataset(Dataset):
    def __init__(self):
        # data = sio.loadmat('data.mat')
        # self.test_x = data['DS_Test']
        # self.test_y = data['yTe']
        trainPath = 'D:\\ZHY\\train.xlsx'
        self.test = pd.read_excel(trainPath).values

    def __len__(self):
        return len(self.test[:,-1])

    def __getitem__(self, idx):
        test = self.test

        """Convert ndarrays to Tensors."""
        return torch.from_numpy(test_x).float()

train_dataset = SBPEstimateDataset()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

total_episodes = EPOCH * len(train_loader)

test_dataset = SBPEDataset()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_te, shuffle=True)



# for step, batch in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
#     train_x = batch['train_x']
#     train_y = batch['train_y']
#     b_x = Variable(train_x)   # batch x
#     b_y = Variable(train_y)
#     print(11111)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=32,  # n_filter
                      kernel_size=9,  # filter size
                      stride=1,  # filter step
                      padding=4,  # con2d出来的图片大小不变
                      ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 1x2采样，o

        )
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 9, 1, 4),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))

        self.conv3 = nn.Sequential(nn.Conv1d(64, 128,  9, 1, 4),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2))
        self.out = nn.Linear(128 * 1 * 32, 3)

    def forward(self, x):
        x = x.view(x.size(0), 1, 256)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5)
optimizer_scheduler = StepLR(optimizer, step_size=int(total_episodes / 5), gamma=0.5)
loss_function = nn.CrossEntropyLoss()


count = 0
test_acc = []
for epoch in range(EPOCH):
    for step, batch in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        optimizer_scheduler.step(count)
        count += 1

        train_x = batch[:,:-1]
        train_y = batch[:,-1]
        b_x = Variable(train_x)  # batch x
        #b_x = b_x.view(b_x.shape[0], b_x.shape[1], 1)
        b_y = Variable(train_y).long().view(train_y.shape[0])
        output = cnn(b_x)

        loss = loss_function(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()


        if step % 50 == 0:
            train_output = cnn(b_x)
            train_y = torch.max(train_output, 1)[1].data.squeeze()
            accuracy_tr = float(sum(train_y == b_y)) / float(b_y.size(0))
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f' % loss.item(), '|train accuracy:%.4f' % accuracy_tr)

            m = 0

            for step_,batch in enumerate(test_loader):
                test_x = batch[:,:-1]
                test_y = batch[:,-1]
                b_tx = Variable(test_x)  # batch x
                # b_x = b_x.view(b_x.shape[0], b_x.shape[1], 1)
                b_ty = Variable(test_y).long().view(test_y.shape[0])
                test_output = cnn(b_tx)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = float(sum(pred_y == b_ty)) / float(test_y.size(0))
                m = accuracy + m
                if step_ % 52 == 0:
                    C = confusion_matrix(b_ty, pred_y)
            m = m / 52
            test_acc.append(m)
            print('|test accuracy:%.4f' % m)
            if max(test_acc) <= m:
                file_name = './output/{}/train_epoch_{}_step_{}.pth'.format(expName, epoch, step)
                torch.save(cnn.state_dict(), file_name)
print('|max accuracy:%.4f' % max(test_acc))
