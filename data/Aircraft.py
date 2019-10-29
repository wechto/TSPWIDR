# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:23:15 2018

@author: Ljx
"""

import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t


class Aircraft(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        a = pd.read_csv('data/us-air-carrier-traffic-statistic.csv')
        b = np.array(a[['Domestic','International']])/1000000
        data = t.from_numpy(b)
        N = data.size()[0]

        ts = np.arange(N)
        ts = t.from_numpy(ts)
        
        self.T, self.future, self.tr_va_te = o.T, o.future, o.tr_va_te
            
        p1, p2 = 0.6, 0.8
        
        move = o.T + o.future - 1
        if self.tr_va_te == 0:
            self.data = data[ : int(N * p1) + move]
            self.ts = ts[ : int(N * p1) + move]
        if self.tr_va_te == 1:
            self.data = data[int(N * p1) : int(N * p2) + move]
            self.ts = ts[int(N * p1) : int(N * p2) + move]
        if self.tr_va_te == 2:
            self.data = data[int(N * p2) : ]
            self.ts = ts[int(N * p2) : ]
                
    def __getitem__(self, index):
        return (t.tensor(self.data[index:index+self.T]), \
                t.tensor(self.data[index + self.T:index + self.T + self.future])), \
                (t.tensor(self.ts[index:index+self.T]), \
                 t.tensor(self.ts[index + self.T:index + self.T + self.future]))
        
    def __len__(self):
        return self.data.size()[0] - self.T - self.future
    
    