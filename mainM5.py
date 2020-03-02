#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:36:28 2020

@author: sergi
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from modelsM5 import M3Model
from P1.DatasetM5 import Datasetm5


def set_parameter_requires_grad(model, toFreeze = False):
    if toFreeze: 
        for param in model.parameters() : 
            param.requires_grad = False

def train():
    
    num_epochs = 50
    batch_size = 150
    
    image_size = 224
    num_classes = 8
    
    freezeInternal = False
    
    #Adaptar el Data Augmentation de com el vam fer al M3
    transform =transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    model = M3Model()
    
    print(batch_size)
    
    set_parameter_requires_grad(model, freezeInternal)
    
    
    TD = Datasetm5()
    dataloader = torch.utils.data.DataLoader(dataset = TD, batch_size = batch_size, shuffle = True, worker_init_fn=1)
    
    VD = Datasetm5()
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
    
    model = model.cuda()
    
    params_to_update = model.parameters()
    
    #Canviar segons lo del M3
    optimizer = optim.SGD(params_to_update, lr = 0.001, momentum = 0.9)
    
    #Utilitzar cross-entropy?
    criterion = nn.CrossEntropyLoss()
    
    x1,lbl = next(iter(dataloader))
    
    x1 = x1.cuda();lbl = lbl.cuda();
    
    val_acc_history = []
    lowest_loss = 9999999999999
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        
        running_loss = 0
        running_corrects = 0
        
        for x,(rinputs, rlabels,rldmrk) in enumerate(dataloader,0) : 
            model.train()
                
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
            print("{}/{} loss : {}".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print('Loss : {:.4f}'.format(epoch_loss))
        
        torch.save(model.state_dict(),'model_weights.pkl')
        
        loss = 0
        for x,(rinputs, rlabels,rldmrk) in enumerate(dataloaderV,0):
            model.eval()    
            inputs = rinputs.cuda()
            labels = rlabels.cuda()
            
            with torch.set_grad_enabled(False) : 
                outputs = model(inputs)
                
                print(outputs)
                print(labels)
                
                loss1 = criterion(outputs, labels)
            
            loss += loss1
        
        loss = loss/(len(dataloader.dataset))
        
        print('Loss : {:.4f}'.format(loss))  
        
        #Accuracy
        
def test():
    
    num_classes = 8
    batch_size = 150
    
    VD = Datasetm5()
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
    
    model = M3Model()
    #Load best model of our model
    model = model.cuda()
    
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    loss = 0
    for x,(rinputs, rlabels,rldmrk) in enumerate(dataloaderV,0):
        model.eval()    
        inputs = rinputs.cuda()
        labels = rlabels.cuda()
            
        with torch.set_grad_enabled(False) : 
            outputs = model(inputs)
                
            print(outputs)
            print(labels)
                
            loss1 = criterion(outputs, labels)
            
        loss += loss1
    
    loss = loss/(len(dataloaderV.dataset))
        
    print('Loss : {:.4f}'.format(loss))
    #Accuracy


train()
test()
    
    