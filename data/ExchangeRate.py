# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 22:55:40 2018

@author: Ljx
"""

import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t


class ExchangeRate(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        a = pd.read_csv('data/exchange_rate.txt', header = None)
#        a = pd.read_csv('exchange_rate.txt', header = None)
        b = np.array(a)
        data = t.from_numpy(b)
        N = data.size()[0]
        
#        print('opt.data_list_index: ', o.data_list_index)
#        data = data[:,o.data_list_index:o.data_list_index+1]

        ts = np.arange(N)
        ts = t.from_numpy(ts)
        
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
        T, future = 800, 5
        train = True
        tr_va_te = 1

    o = O()
    er = ExchangeRate(o)
    er_data, er_ts = er[0]
    print(er[0][0][0].size())
    print(er[0][0][1].size())

    data = er_data[0][:,0].numpy()
    import numpy as np
    from scipy.fftpack import fft,ifft
    import matplotlib.pyplot as plt
    import seaborn
    y = data
    x = np.linspace(0, 1, len(y))
    
    ylimax = 0.002
    plt_f_lim = int(len(x)/2)
    ffty = fft(y)/len(x)
    plt.subplot(3,2,1);
    plt.plot(x, y)
#    plt.show()
    plt.subplot(3,2,2);
    plt.plot(x[0:plt_f_lim], ffty[0:plt_f_lim]);plt.ylim([0,ylimax])
    plt.show()
    ffty_ = ffty
#    ffty_[abs(ffty_)<0.3] = 0
    iffty_ = ifft(ffty_)
    plt.subplot(3,2,5);
    plt.plot(x, iffty_)
#    plt.show()
    plt.subplot(3,2,6);
    plt.plot(x[0:plt_f_lim], ffty_[0:plt_f_lim]);plt.ylim([0,ylimax])
    plt.show()