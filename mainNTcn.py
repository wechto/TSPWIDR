# -*- coding: utf-8 -*-


import models, data
from utils.visualize import Visualizer 
from config import opt
import torch as t, numpy as np
import warnings
from torch.utils.data import DataLoader

def train(f='train'):
    if f=='train':
        opt.tr_va_te = 0
    # model
    model = getattr(models, opt.model)(opt = opt)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    
    model.train()
    # data
    train_data = data.datasets(opt).getData()
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle = True)
    # certerion & optimzer
    certerion = t.nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(),
                             lr = opt.lr, 
                             weight_decay = opt.weight_decay)
    # losses
    epoch_losses_train = [] # np.zeros(opt.max_epoch)
    epoch_losses_test = []

    #train
    for epoch in range(opt.max_epoch):
        temp_loss_train = []
        for _, (d_t, ts) in enumerate(train_dataloader):
            input_data = d_t[0].to(opt.device).permute(1,0,2)
            target_data = d_t[1].to(opt.device).permute(1,0,2)
            _temp_loss = []
            ifft_i = input_data
            for recurrent_i in range(opt.recurrent_n+1):
                optimizer.zero_grad()
                output_data, ifft_i, ifft_t = model(ifft_i, target_data, recurrent_i)
                loss_train = 0.9 * certerion(ifft_t, output_data) + 0.1 * certerion(ifft_i, input_data)
                loss_train.backward()
                optimizer.step()
                _temp_loss.append(loss_train.item())
                ifft_i = ifft_i.detach()
            temp_loss_train.append(_temp_loss)

        epoch_losses_train.append(np.array(temp_loss_train).mean(axis = 0))
        # TODO(ljx): print
        if epoch % 50 == 0:
            with t.no_grad():
                model.eval()
                _, pre_out = pre_(model)
            model.train()
            print(epoch, np.round(np.array(epoch_losses_train[-1])), np.round(np.array(pre_out[0])),
                  np.round(np.array(pre_out[1]),4), np.round(np.array(pre_out[2]),4), opt.lr)
        if epoch < 5:
            continue
        if (epoch_losses_train[-1] > epoch_losses_train[-2]).sum() == opt.recurrent_n+1 :
            opt.lr = opt.lr * opt.lr_decay
        if epoch % 1000 == 0:
            model.save(opt, name = str(epoch)+'_'+str(opt.data_list_index))

    path = model.save(opt, name = str(opt.data_list_index))
    return (epoch_losses_train, epoch_losses_test, path), None

    

def pre(f='pre'):
    opt.tr_va_te = opt.train_test_pre
    
    model = getattr(models, opt.model)(opt = opt).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
#    print('\n\n', list(model.out_sigma.named_parameters()), '\n\n')
    return pre_(model)
    
def pre_(model):
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
        
        _temp_loss = []
        ifft_i = input_data
        output_dataN = []
        for recurrent_i in range(opt.recurrent_n + 1):
            output_data, ifft_i, ifft_t = model(ifft_i, target_data, recurrent_i)
            loss_train = 0.9 * t.nn.MSELoss()(ifft_t, output_data) + 0.1 * t.nn.MSELoss()(ifft_i, input_data)
            _temp_loss.append(loss_train.item())
            output_dataN.append(output_data.cpu().detach().numpy())
            ifft_i = ifft_i.detach()
        loss.append(_temp_loss)
        
        def tensor2numpy(i_, o_, t_):
            return i_.cpu().detach().numpy(), \
                o_.cpu().detach().numpy(), \
                t_.cpu().detach().numpy()
        i_, o_, t_ = tensor2numpy(input_data, t.from_numpy(np.array(output_dataN)), target_data)
        all_input_data = i_ if all_input_data is None else np.concatenate([all_input_data, i_])
        all_output_data = o_ if all_output_data is None else np.concatenate([all_output_data, o_], axis = 1)
        all_target_data = t_ if all_target_data is None else np.concatenate([all_target_data, t_])
        all_ts = ts if all_ts is None else (t.cat((all_ts[0], ts[0]), dim = 0), t.cat((all_ts[1], ts[1]), dim = 0))
        index += opt.future
#    print(all_input_data.shape, all_output_data.shape, all_target_data.shape)
    all_ts = (all_ts[0].unsqueeze(0), all_ts[1].unsqueeze(0))
    Visualizer().drawTestN(([], all_output_data, all_target_data), all_ts, drawLot = True)
    MSE, U = evaluation(t.from_numpy(all_output_data), t.from_numpy(all_target_data), 0)
    return None,( np.array(loss).mean(axis=0), MSE, U)
    
def evaluation(outputN, target, batch):
    mse, u = [], []
    for i in range(outputN.shape[0]):
        output = outputN[i]
        numerator = t.sqrt(t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)))
        denominator_u = t.sqrt(t.mean(t.pow(target[:,batch,:],2))) * t.sqrt(t.mean(t.pow(output[:,batch,:],2)))
        MSE = t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)).item()
        U = t.div(numerator, denominator_u).item()
        mse.append(MSE)
        u.append(U)
    return mse, u

def help():
    pass

def LetsGo(kwargs, fun):
    if kwargs is not None:
        opt._parse(kwargs)
    opt.input_size = opt._input_kv[opt.data]
    opt.output_size = opt._output_kv[opt.data]
    opt.needLog = opt._needLog_kv[opt.data]
    print(opt.model, '\tN:', opt.recurrent_n, '\tData:',opt.data_list_index)
    out_train, out_pre = fun()
    if out_train:
        print('path:', out_train[2])
        if len(out_train[0]) > 10:
            if out_train[0]:
                print('epoch_losses_train')
                Visualizer().drawEpochLossN(out_train[0][10:])
            if out_train[1]:
                print('epoch_losses_test')
                Visualizer().drawEpochLossN(out_train[1][10:])
    if out_pre:
        print('loss:',out_pre[0], '\nMSE:', out_pre[1], '\nU:', out_pre[2])
    return out_train, out_pre


if __name__ == '__main__':
    t.set_default_tensor_type('torch.DoubleTensor')
#    m_path = 'checkpoints/RecurrentNTcn_NINO_1000_1_0418063025.pth'
    m_path = None
    m_model = 'RecurrentNTcn' 
    m_data = 'NINO' # NINO, Yahoo, Wecar, Aircraft, BJpm
    m_lr = 1.0e-5
    m_num = 80
    
    opt.max_epoch = 2000
#    opt.data_list_index = 1
    opt.data_list_index = 2 #tqx
    opt.recurrent_n = 6
    opt.model_list_path[1] = 'checkpoints/Seq2Seq_NINO_0123193337.pth'
     
    mm = {'load_model_path':m_path, 'model':m_model, 'data':m_data, 
          'lr':m_lr, 'num':m_num}
    
    out_train, out_pre = LetsGo(mm, train) # train, pre
    
    