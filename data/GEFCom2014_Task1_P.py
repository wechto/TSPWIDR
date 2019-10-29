# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:53:20 2018

@author: Ljx
"""


import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t



class GEFCom2014_Task1_P(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        a = pd.read_csv('data/GEFCom2014_Task1_P.csv')
#        a = pd.read_csv('GEFCom2014_Task1_P.csv')
        
        b = np.array(a[['Zonal Price']])
        data = t.from_numpy(b)
        N = data.size()[0]

        ts = np.arange(N)
        ts = t.from_numpy(ts)
        data = data[:,0:1]
        
        self.T, self.future, self.tr_va_te = o.T, o.future, o.tr_va_te
            
        p1, p2 = 0.78, 0.8
        
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
        return (self.data[index:index+self.T].clone().detach(), \
                self.data[index + self.T:index + self.T + self.future].clone().detach()), \
                (self.ts[index:index+self.T].clone().detach(), \
                 self.ts[index + self.T:index + self.T + self.future].clone().detach())
        
    def __len__(self):
        return self.data.size()[0] - self.T - self.future

if __name__ == '__main__':
    print(os.getcwd())
    class O():
        T, future = 200, 5
        train = True
        tr_va_te = 0

    o = O()
    GEFCom2014 = GEFCom2014_Task1_P(o)
    print(GEFCom2014.__len__())
    GEFCom2014_data, GEFCom2014_ts = GEFCom2014[900]
    print(GEFCom2014_data[1])
    print(GEFCom2014_ts[1])
    
    data = GEFCom2014_data[0][:,0].numpy()
    import numpy as np
    from scipy.fftpack import fft,ifft
    import matplotlib.pyplot as plt
    import seaborn
    y = data
#    print(y)
    x = np.linspace(0, 100, len(y))    
    plt.plot(x, y)
    
    