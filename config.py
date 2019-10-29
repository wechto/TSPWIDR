# -*- coding: utf-8 -*-

import warnings
import torch as t


class DefaultConfig(object):
    
    model = 'LSTM'
    data = 'NINO3'
    
    load_model_path = None
    
    input_size, output_size = None, None
    needLog = None
    tr_va_te = None
    encoder_hidden_size = 64  # 64,128,256
    decoder_hidden_size = 64  # 64,128,256 
    
    batch_size = 16 # 7, 3
    T, future = 64, 4
    use_gpu = True
    print_freq = 20
    
    model_list_path = [None, None, None, None] # 0:aucoencoder   1:seq2seq   2:recurrent  3:fusion
    train_test_pre = 2
    data_list_index = 0
    recurrent_n = 1
    
    max_epoch = 2500

    lr = 0.01 # initial learning rate
    lr_decay = .95 # when val_loss incress, lr = lr * lr_dacay
    weight_decay = 1e-5
    
    num = 15 # for test & val

    
    device = t.device('cuda') if use_gpu else t.device('cpu')
    
    _input_kv = {'ExchangeRate':8, 'NINO':1, 'Yahoo':1, 'Wecar':5, 'Aircraft':2, 'BJpm':4,'GEFCom2014_Task1_P':1}
    _output_kv = {'ExchangeRate':8, 'NINO':1, 'Yahoo':1, 'Wecar':5, 'Aircraft':2, 'BJpm':4,'GEFCom2014_Task1_P':1}
    _needLog_kv = {'ExchangeRate':False, 'NINO':True, 'Yahoo':True, 'Wecar':False, 'Aircraft':True,  'BJpm':False,'GEFCom2014_Task1_P':True}
    
    def _parse(self, kwargs = {}, printconfig = False):
        '''
        更新参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' % k)
            setattr(self, k, v)
        if printconfig:
            print('\nuser config:')
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(k, getattr(self, k))
        

opt = DefaultConfig()

if __name__ == '__main__':
    opt._parse(printconfig = True)
    print(opt.lr)

    
    
