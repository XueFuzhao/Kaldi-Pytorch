#!_tf) bash-4.2$ vim usr/bin/python3

##  ##  Copyright (C) 2017 Huang Hengguan
##  hhuangaj [at] ust [dot] hk
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


from dataGenSequences_cxt_nolap_fbank import dataGenSequences
from compute_priors import compute_priors
from shutil import copy
import sys
import os
import torch.utils.data as data
import torch.nn as nn
import torch
from lib.ops import Dense,NaiveLSTM

##!!! please modify these hyperprameters manually

# this depend on the feature you applied
mfccDim=23


if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) != 8:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir dnn_dir init_lr')


data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
exp     = sys.argv[6]
init_lr = float(sys.argv[7])

##!!! please modify these hyperprameters manually
## Learning parameters
learning = {'rate' : init_lr,
            'singFeaDim' : mfccDim, 
            'minEpoch' : 30,
            'batchSize' : 256,#40 at first
            'timeSteps' : 20,
            'dilDepth' : 1,
            'minValError' : 0,
            'left' : 0,
            'right': 4,
            'hiddenDim' : 1024,
            'modelOrder' : 1,
            'layerNum': 12,
            'historyNum' : 1}

## Copy final model and tree from GMM directory
os.makedirs (exp, exist_ok=True)
copy (gmm + '/final.mdl', exp)
copy (gmm + '/tree', exp)

## Compute priors
compute_priors (exp, ali_tr, ali_cv)

# The input feature of the neural network has this form:  0-1-4 features
feaDim = (learning['left'] + learning['right']+1)*mfccDim

# load data from data iterator
trDataset = dataGenSequences (data_tr, ali_tr, gmm,learning['timeSteps'], feaDim,learning['left'],learning['right'])
cvDataset = dataGenSequences (data_cv, ali_cv, gmm,learning['timeSteps'], feaDim,learning['left'],learning['right'])

# Recommend shuffle=False, because this iterator's shuffle can only work on the single split
trGen = data.DataLoader(trDataset,batch_size=learning['batchSize'],shuffle=False,num_workers=0)
cvGen = data.DataLoader(cvDataset,batch_size=learning['batchSize'],shuffle=False,num_workers=0)


##load the configurations from the training data
learning['targetDim'] = cvDataset.outputFeatDim

class lstm(nn.Module):
    def __init__(self,input_size = feaDim, hidden_size = 1024 , output_size = 1095 ):
        super(lstm,self).__init__()
        self.Dense_layer1 = nn.Sequential(Dense(input_size,hidden_size))
        self.lstm_layer2 = nn.Sequential(nn.LSTM(hidden_size, hidden_size,num_layers=3))
        self.Dense_layer3 = nn.Sequential(Dense(hidden_size, output_size))

    def forward(self,x):
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer1(x)
        x = torch.reshape(x, (b, t, 1024))
        x,h = self.lstm_layer2(x)
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer3(x)
        return x


# The following two classes are the modules I tried
'''class lstm(nn.Module):
    def __init__(self,input_size = feaDim, hidden_size = 1024 , output_size = 1095 ):
        super(lstm,self).__init__()
        self.Dense_layer1 = nn.Sequential(Dense(input_size,1024))
        self.lstm_layer2 = nn.Sequential(NaiveLSTM(1024, 1024))
        self.lstm_layer3 = nn.Sequential(NaiveLSTM(1024, 1024))
        self.lstm_layer4 = nn.Sequential(NaiveLSTM(1024, 2048))
        self.Dense_layer5 = nn.Sequential(Dense(2048, output_size))

    def forward(self,x):
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer1(x)
        x = torch.reshape(x, (b, t, 1024))
        x,h = self.lstm_layer2(x)
        x, h = self.lstm_layer3(x)
        x, h = self.lstm_layer4(x)
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer5(x)
        return x'''

'''class lstm(nn.Module):
    def __init__(self,input_size = feaDim, hidden_size = 1024 , output_size = 1095 ):
        super(lstm,self).__init__()
        self.lstm_layer1 = nn.Sequential(Dense(input_size,1024))
        self.lstm_layer2 = nn.Sequential(Dense(512,2048))
        self.lstm_layer3 = nn.Sequential(Dense(2048, 4096))
        self.lstm_layer4 = nn.Sequential(Dense(4096,output_size))

    def forward(self,x):
        x = self.lstm_layer1(x)
        x = self.lstm_layer2(x)
        x = self.lstm_layer3(x)
        x = self.lstm_layer4(x)
        return x'''


# If you run this code on CPU, please remove the '.cuda()'
model = lstm(input_size=feaDim, hidden_size= learning['hiddenDim'],output_size=learning['targetDim']).cuda()

loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD( model.parameters(),lr=0.25,momentum=0.5,weight_decay=0.001)
optimizer = torch.optim.Adam( model.parameters(),lr=learning['rate'],weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)


def train(model, train_loader, my_loss, optimizer, epoch):
    model.train()
    acc = 0
    for batch_idx, (x,y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        '''batch_num, time_steps, hidden_size = output.size()
        output = torch.reshape(output,(batch_num*time_steps, hidden_size))'''
        y_batch_size, y_time_steps = y.size()
        y = torch.reshape(y,tuple([y_batch_size*y_time_steps]))
        y = y.long()
        loss = my_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output.data, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        if (batch_idx % 10 == 0):
            print("train:        epoch:%d ,step:%d, loss:%f"%(epoch+1,batch_idx,loss))

    print(acc)
    print(cvDataset.numFeats)
    print("Accuracy: %f"%(acc/cvDataset.numFeats))


def val(model, train_loader, my_loss, optimizer, epoch):
    model.train()
    acc = 0
    for batch_idx, (x,y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        output = model(x)
        '''batch_num, time_steps, hidden_size = output.size()
        output = torch.reshape(output,(batch_num*time_steps, hidden_size))'''
        y_batch_size, y_time_steps = y.size()
        y = torch.reshape(y,tuple([y_batch_size*y_time_steps]))
        y = y.long()
        loss = my_loss(output, y)
        _, pred = torch.max(output.data, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        if (batch_idx % 10 == 0):
            print("val:        epoch:%d ,step:%d, loss:%f"%(epoch+1,batch_idx,loss))
    print(acc)
    print(cvDataset.numFeats)
    print("Accuracy: %f"%(acc/cvDataset.numFeats))

for epoch in range(9000):
    print("=====================================================================")
    train(model, trGen, loss_function, optimizer, epoch)
    scheduler.step()
    print(scheduler.get_lr())
    print("===========================")
    if epoch % 5 == 0 :
        val(model, cvGen, loss_function, optimizer, epoch)


