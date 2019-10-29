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
#        print('model : ',opt.model,', epoch : ',epoch)
        temp_loss_train = []
        for _, (d_t, ts) in enumerate(train_dataloader):
#            print(_, end=' ')
            input_data = d_t[0].to(opt.device)
            target_data = d_t[1].to(opt.device)
            
            optimizer.zero_grad()
            input_data = input_data.permute(1,0,2)
            target_data = target_data.permute(1,0,2)
#            print(input_data.size(), target_data.size())
            output_data, output_former = model(input_data)
            loss_train = certerion(input_data, output_data)
            loss_train.backward()
            optimizer.step()
            temp_loss_train.append(loss_train.item())

        epoch_losses_train.append(np.mean(temp_loss_train))
        # TODO(ljx): print
        if epoch % 50 == 25:
            print('model:',opt.model,' ,epoch:',epoch, 'train loss:',epoch_losses_train[-1], opt.lr)
        if epoch < 3:
            continue
        if epoch_losses_train[-1] > epoch_losses_train[-2]:
            opt.lr = opt.lr * opt.lr_decay

    path = model.save(opt)
    return epoch_losses_train, epoch_losses_test, path

    

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
        output_data, _ = model(input_data)
#        print(input_data.shape, target_data.shape,output_data.shape)
        temp_loss = t.nn.MSELoss()(input_data, output_data).item()
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
    print(all_input_data.shape, all_output_data.shape, all_target_data.shape)
    all_ts = (all_ts[0].unsqueeze(0), all_ts[1].unsqueeze(0))
#    Visualizer().drawTest(([], all_output_data, all_input_data), all_ts, drawLot = True)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(all_output_data[:100, 0, 0])
    plt.plot(all_input_data[:100, 0, 0])
    plt.show()
    evaluation(t.from_numpy(all_output_data), t.from_numpy(all_input_data), 0)
    return loss, None, None
    
def evaluation(output, target, batch):
    numerator = t.sqrt(t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)))
    denominator_u = t.sqrt(t.mean(t.pow(target[:,batch,:],2))) * t.sqrt(t.mean(t.pow(output[:,batch,:],2)))
    MSE = t.mean(t.pow(output[:,batch,:]-target[:,batch,:],2)).item()
    U = t.div(numerator, denominator_u).item()
    print('model: ',opt.model)
    print('MSE: ',MSE)
    print('U: ',U)

def help():
    pass

def LetsGo(kwargs, fun):
    if kwargs is not None:
        opt._parse(kwargs)
    opt.input_size = opt._input_kv[opt.data]
    opt.output_size = opt._output_kv[opt.data]
    opt.needLog = opt._needLog_kv[opt.data]
    epoch_losses_train, epoch_losses_test, path = fun()
    print('path : ',path)
    if len(epoch_losses_train) > 1:
        if epoch_losses_train:
            print('epoch_losses_train')
            Visualizer().drawEpochLoss(epoch_losses_train[10:])
#        if epoch_losses_test:
#            print('epoch_losses_test')
#            Visualizer().drawEpochLoss(epoch_losses_test[10:])
#    print('\n','epoch_losses_train:\n',epoch_losses_train)
#    print('\n','epoch_losses_train:\n',epoch_losses_test)
    return path



if __name__ == '__main__':
    t.set_default_tensor_type('torch.DoubleTensor')
    
    '''
    AutoEncoder_NINO_0116113812, AutoEncoder_NINO_0117104946
    '''
    m_path = 'checkpoints/AutoEncoder_NINO_0122203529.pth' 
    m_path = None
    
    '''
    multi
    '''
    m_model = 'AutoEncoder' 
    m_data = 'NINO' # NINO, Yahoo, Wecar, Aircraft, BJpm
    m_lr = 0.001
    m_num = 80
    
    mm = {'load_model_path':m_path, 'model':m_model, 'data':m_data, 
          'lr':m_lr, 'num':m_num}
    
    LetsGo(mm, pre) # train, pre
    
    opt._parse(printconfig = True)
    
def main_run(run_opt, run_data):
    t.set_default_tensor_type('torch.DoubleTensor')
    print('\nAutoEncoder')
    opt = run_opt
    m_path = None
    m_model = 'AutoEncoder' 
    m_data = run_data # NINO, Yahoo, Wecar, Aircraft, BJpm
    m_lr = 0.001
    m_num = 80
    mm = {'load_model_path':m_path, 'model':m_model, 'data':m_data, 
          'lr':m_lr, 'num':m_num}
    
    newpath = LetsGo(mm, train)
    opt._parse(printconfig = True) 
    return newpath

def main_run_pre(run_opt, run_data):
    t.set_default_tensor_type('torch.DoubleTensor')
    print('\nAutoEncoder')
    opt = run_opt
    m_path = opt.model_list_path[0]
    m_model = 'AutoEncoder' 
    m_data = run_data # NINO, Yahoo, Wecar, Aircraft, BJpm
    m_lr = 0.001
    m_num = 80
    mm = {'load_model_path':m_path, 'model':m_model, 'data':m_data, 
          'lr':m_lr, 'num':m_num}
    
    newpath = LetsGo(mm, pre)
#    opt._parse(printconfig = True) 
    return newpath
    
    
    
