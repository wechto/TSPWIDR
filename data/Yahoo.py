# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:34:05 2018

@author: Ljx
"""

import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t


class Yahoo(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        a = pd.read_csv('data/yahoo_open_6.csv')
        b = np.array(a[['JPM','GS','BAC','C','WFC','MS']])
        data = t.from_numpy(b)
        N = data.size()[0]

        ts = np.arange(N)
        ts = t.from_numpy(ts)
        
        print('opt.data_list_index: ', o.data_list_index)
        data = data[:,o.data_list_index:o.data_list_index+1]
        
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
        T, future = 20, 5
        train = True
        tr_va_te = 1

    o = O()
    yahoo = Yahoo(o)
    print(yahoo[0][0][0].size())
    print(yahoo[0][0][1].size())
    
# =============================================================================
# import numpy as np, pandas as pd
# from pandas_datareader import DataReader
# from datetime import datetime
# 
# 
# data = DataReader(['JPM','GS','BAC','C','WFC','MS'],  'yahoo', datetime(2012,1,1), datetime(2016,6,28)) #datetime(2007,1,2), datetime(2018,12,25)
# data = np.log(data['High'])
# data.to_csv('yahoo_open_6.csv')
# data = None
# 
# data = pd.read_csv('yahoo_open_6.csv')
# print(data)
# =============================================================================
