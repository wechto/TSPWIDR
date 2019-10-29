# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:59:39 2019

@author: Ljx
"""

from .BasicModule import BasicModule
from .AutoEncoder import AutoEncoder
from .Recurrent import Recurrent

import torch as t
import torch.nn as nn


class multiF(BasicModule):
    def __init__(self, opt):      
        super(multiF, self).__init__()

        self.opt = opt
        
        self.autoencoder = AutoEncoder(self.opt)

        self.autoencoder.load(opt.model_list_path[0])

        self.recurrent = Recurrent(self.opt)
        self.recurrent.load(opt.model_list_path[2])
        
#        self.down_fusion = nn.Linear(self.opt.T, 32)
        

        self.fusion_1_up = nn.Linear(16, self.opt.future)
        self.fusion_1_down = nn.Linear(self.opt.future, self.opt.future)

        self.fusion_2_up = nn.Linear(16, self.opt.future)
        self.fusion_2_down = nn.Linear(self.opt.future, self.opt.future)
        
        self.fusion_out_1 = nn.Linear(self.opt.future, self.opt.future)
        self.fusion_out_2 = nn.Linear(self.opt.future, self.opt.future)
    
    def forward(self, input_data, target_data):
        self.autoencoder.load(self.opt.model_list_path[0])

        self.recurrent.load(self.opt.model_list_path[2])
        
        _, up = self.autoencoder(input_data)

        down, _, _ = self.recurrent(input_data, target_data)
        down = down.squeeze(2).permute(1, 0)
        
        down = self.fusion_1_up(up) + self.fusion_1_down(down)
        down = t.nn.ReLU()(down)
        down = self.fusion_2_up(up) + self.fusion_2_down(down)
#        middle = t.tanh(middle)
        out = self.fusion_out_2(self.fusion_out_1(down))
        return out.permute(1,0).unsqueeze(2)
    