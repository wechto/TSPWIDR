# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:26:12 2018

@author: Ljx
"""

import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t


class BJpm(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        a = pd.read_csv('data/BJ-PM.csv')
        b = np.array(a[['pm2.5','DEWP','TEMP','PRES']]) # 'pm2.5','DEWP','TEMP','PRES','Iws'
        b = self._clearNan(b)
        b = self._ddd(b)
        data = t.from_numpy(b)

        N = data.size()[0]

        ts = np.arange(N)
        ts = t.from_numpy(ts)
        
        self.T, self.future, self.tr_va_te = o.T, o.future, o.tr_va_te
            
        p1, p2 = 0.1, 0.8
        
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
    
    
    def _clearNan(self,data):
        asdf = data.shape[0]
        aassddff = data.shape[1]
        
        b = np.isnan(data[...,0]).reshape(-1,1)
        c = np.repeat(b, aassddff, axis = 1)
        
        asdf_ = np.sum(c,axis = 0)[0]
        
        c = list(~c.reshape(-1))
        
        data = data.reshape(-1)
        
        d = data[c]
        
        d = d.reshape(asdf-asdf_, aassddff)
        
        return d

    def _ddd(self, data):
        data[..., 3] = data[..., 3] - 1000
        return data
    
if __name__ == '__main__':
    print(os.getcwd())
    class O():
        T, future = 20, 5
        train = True
        tr_va_te = 1

    o = O()
    bjpm = BJpm(o)
    print(bjpm[0][0][0].size())
    print(bjpm[0][0][1].size())