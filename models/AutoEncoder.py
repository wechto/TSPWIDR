# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 20:38:12 2019

@author: Ljx
"""

from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import numpy as np


class AutoEncoder(BasicModule):
    def __init__(self, opt):      
        super(AutoEncoder, self).__init__()
        self.module_name = 'AutoEncoder'
        self.opt = opt
        
        self.d_1 = nn.Linear(self.opt.T, 48)
        self.d_2 = nn.Linear(48, 32)
        self.d_3 = nn.Linear(32, 16)
        
        self.u_1 = nn.Linear(16, 32)
        self.u_2 = nn.Linear(32, 48)
        self.u_3 = nn.Linear(48, self.opt.T)
    
    def forward(self, x):
        x = x.squeeze(2).permute(1, 0)
        x = self.d_1(x)
        x = self.d_2(x)
        x = self.d_3(x)
        out = x.detach()
        x = self.u_1(x)
        x = self.u_2(x)
        x = self.u_3(x)
        x = x.permute(1, 0).unsqueeze(2)
        return x, out
