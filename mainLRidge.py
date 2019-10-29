# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge,RidgeCV

import data
from utils.visualize import Visualizer 
from config import opt
import torch as t
import warnings
from torch.utils.data import DataLoader


def train():
    opt.tr_va_te = 0
    opt.batch_size = 1
    # data
    train_data = data.datasets(opt).getData()
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle = True)
    #train
    for _, (d_t, ts) in enumerate(train_dataloader):
        input_data = d_t[0].to(opt.device)
        target_data = d_t[1].to(opt.device)

        input_data = input_data.permute(1,0,2).squeeze(2).permute(1,0)
        target_data = target_data.permute(1,0,2).squeeze()[0:1]
#        print(input_data.size(), target_data.size())
        X = input_data if _ == 0 else t.cat([X, input_data], dim = 0)
        y = target_data if _ == 0 else t.cat([y, target_data], dim = 0)
    X, y = X.cpu().numpy(), y.cpu().numpy()
#    print('shape: ',X.shape,y.shape)
    return X, y

def pre_(model):
    opt.tr_va_te = 2
    test_data = data.datasets(opt).getData()
    N = test_data.__len__()
    if opt.num > N :
        warnings.warn('Warning: data is not long enough, data(%d) num(%d)' % (N, opt.num))
    start_index = 0
    all_input_data = None
    all_output_data = None
    all_target_data = None
    all_ts = None
    loss = []
    index = start_index
    while index - start_index < opt.num:
        def datatype(test_data, index):
            d_t, ts = test_data[index]
            i = d_t[0].to(opt.device).unsqueeze(0).permute(1,0,2)
            t = d_t[1].to(opt.device).unsqueeze(0).permute(1,0,2)
            return i, t, ts
        input_data, target_data, ts = datatype(test_data, index)
        output_data = model_LRidge(model, input_data, target_data)
#        print(input_data.shape, target_data.shape,output_data.shape)
        temp_loss = t.nn.MSELoss()(target_data, output_data)#/opt.future + t.nn.MSELoss()(ifft_i, input_data)/opt.T
        loss.append(temp_loss)
        def tensor2numpy(i_, o_, t_):
            return i_.cpu().detach().numpy(), \
                o_.cpu().detach().numpy(), \
                t_.cpu().detach().numpy()
        i_, o_, t_ = tensor2numpy(input_data, output_data, target_data)
        all_input_data = i_ if all_input_data is None else np.concatenate([all_input_data, i_])
        all_output_data = o_ if all_output_data is None else np.concatenate([all_output_data, o_])
        all_target_data = t_ if all_target_data is None else np.concatenate([all_target_data, t_])
        all_ts = ts if all_ts is None else (t.cat((all_ts[0], ts[0]), dim = 0), t.cat((all_ts[1], ts[1]), dim = 0))
        index += opt.future
#    print(all_input_data.shape, all_output_data.shape, all_target_data.shape)
    all_ts = (all_ts[0].unsqueeze(0), all_ts[1].unsqueeze(0))
#    Visualizer().drawTest(([], all_output_data, all_target_data), all_ts, drawLot = True)
    evaluation(t.from_numpy(all_output_data), t.from_numpy(all_target_data), 0)
    return loss, None, None
    
def evaluation(output, target, batch):
    numerator = t.sqrt(t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)))
    denominator_u = t.sqrt(t.mean(t.pow(target[:,batch,:],2))) * t.sqrt(t.mean(t.pow(output[:,batch,:],2)))
    MSE = t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)).item()
    U = t.div(numerator, denominator_u).item()
    print(opt.T, ' -- ',opt.future)
    print('MSE: ',MSE)
    print('U: ',U)

def model_LRidge(model, input_data, target_data):
    input_data = input_data.squeeze(2).permute(1, 0).cpu().numpy()
    
    for i in range(opt.future):
        y = model.predict(input_data if i==0 else t.cat([t.from_numpy(input_data)[:, i:], ys.unsqueeze(0)],dim=1).numpy())
        ys = t.from_numpy(y) if i==0 else t.cat([ys, t.from_numpy(y)], dim = 0)
#    print(ys.shape)
    return ys.unsqueeze(1).unsqueeze(2).to(opt.device)

def onlyone():
    opt.data_list_index = 3
    opt.num = 80
    X, y = train()
    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    model.fit(X, y)
    pre_(model)
    
def batch():
    opt.num = 80
    T_future = [[64,2], [64,4], [64,8]]
    for i in range(1):
        opt.data_list_index = i
        for j in range(3):
            opt.T, opt.future = T_future[j][0], T_future[j][1]
            X, y = train()
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
            model.fit(X, y)
            pre_(model)
        print('')

if __name__ == '__main__':
    print('model: ','LRidge')
    opt.data = 'GEFCom2014_Task1_P'
#    onlyone()
    batch()
    
    
    
    