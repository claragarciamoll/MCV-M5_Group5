import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime

class PokeNet(nn.Module):
    def __init__(self):
        super(PokeNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=4),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3,stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(3,stride=2))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1),
                                    nn.ReLU())

        self.gap = nn.Sequential(nn.AvgPool2d(6), nn.Softmax())

    def forward(self, image):
        out1 = self.layer1(image)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        out4 = self.conv1(out3)

        out = self.gap(out4)
        out = out.squeeze()

        return out


def main():
    image_size = 224
    
    multi_gpu = False

    num_classes = 8
    num_epochs = 120
    batch_size = 32
    batch_size_val = 64

    train_data_dir = '/home/mcv/datasets/MIT_split/train'
    test_data_dir = '/home/mcv/datasets/MIT_split/test'  

    transform = {'train': transforms.Compose([transforms.Resize((image_size,image_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(5),
                                    transforms.ToTensor()]),
                'test': transforms.Compose([transforms.Resize((image_size,image_size)),
                                    transforms.ToTensor()])}


    model = PokeNet()

    print(model)
    print('batch_size : ',batch_size)


    if multi_gpu :
        batch_size*=2

    #Dataloaders
    dataset = torchvision.datasets.ImageFolder(train_data_dir, transform=transform['train'])
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*.8)+1, int(len(dataset)*.2)])

    dataloader = torch.utils.data.DataLoader(dataset = train_set.dataset, batch_size = batch_size, shuffle = True,num_workers=16)
    dVD = torch.utils.data.DataLoader(dataset = val_set.dataset, batch_size = batch_size_val, shuffle = False,num_workers=16)

    dataset_test = torchvision.datasets.ImageFolder(test_data_dir, transform=transform['test'])
    dTD = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = batch_size, num_workers=16)

    # Loss & optimizer
    optimizer = optim.Adadelta(list(model.parameters()))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=0.0001, verbose=True)

    criterion = nn.CrossEntropyLoss()
    
    ## Training ##

    model.cuda()
    writer = SummaryWriter()

    for epoch in range(num_epochs) :
        print('-'*10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        running_loss = 0
        corrects = 0
        
        model.train()

        for x, (rinputs, rlabels) in enumerate(dataloader,0):            

            inputs = rinputs.cuda()
            labels = rlabels.cuda()
            
            #zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True) : 
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            
            running_loss += loss.item() * inputs.size(0) 
            values, indices = torch.max(outputs, 1)                
            correct = (indices == labels).float().sum()
            corrects += correct            
            
        epoch_loss = running_loss / len(dataloader.dataset)
        acc = 100*corrects/len(dataloader.dataset)
        print('Train - Loss : {:.4f} Accuracy: {}'.format(epoch_loss,acc))

        ## Validation ##
        running_loss_val = 0
        corrects_val = 0

        model.eval()

        for x, (rinputs, rlabels) in enumerate(dVD, 0):

            inputs = rinputs.cuda()
            labels = rlabels.cuda()

            with torch.set_grad_enabled(False):

                outputs = model(inputs)

                loss_val = criterion(outputs, labels)
            
            running_loss_val += loss_val.item() * inputs.size(0)

            values, indices = torch.max(outputs, 1)
            correct_val = (indices == labels).float().sum()
            corrects_val += correct_val
            
        epoch_loss_val = running_loss_val / len(dVD.dataset)
        scheduler.step(epoch_loss_val)

        acc_val = 100*corrects_val/len(dVD.dataset)
        print('Val   - Loss : {:.4f} Accuracy: {}'.format(epoch_loss_val,acc_val))

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', epoch_loss_val, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Accuracy/val', acc_val, epoch)

    # Save model
    torch.save(model.state_dict(),'model_weights.pkl')

    ## Test ##

    running_loss_test = 0
    corrects_test = 0

    model.eval()

    for x, (rinputs, rlabels) in enumerate(dTD, 0):

        inputs = rinputs.cuda()
        labels = rlabels.cuda()

        with torch.set_grad_enabled(False):

            outputs = model(inputs)

            loss_test = criterion(outputs, labels)
        
        running_loss_test += loss_test.item() * inputs.size(0)

        values, indices = torch.max(outputs, 1)
        correct_test = (indices == labels).float().sum()
        corrects_test += correct_test
        
    loss_test = running_loss_test / len(dTD.dataset)

    acc_test = 100*corrects_test/len(dTD.dataset)
    print('Val   - Loss : {:.4f} Accuracy: {}'.format(loss_test,acc_test))

main()
