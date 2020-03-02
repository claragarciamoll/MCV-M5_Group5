#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:29:42 2020

@author: sergi
"""

#MODELS

import torch
import torch.nn as nn

class M3Model(nn.Module):
    def __init__(self):
        super(M3Model, self).__init__()
        
    def forward(self,x):
        return x
    
