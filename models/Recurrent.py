# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:20:34 2019

@author: Ljx
"""

from .BasicModule import BasicModule
from .Seq2Seq import Seq2Seq

import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

class Recurrent(BasicModule):
    def __init__(self, opt):      
        super(Recurrent, self).__init__()
        self.module_name = 'Recurrent'
        self.opt = opt

        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        
        self.seq2seq = Seq2Seq(self.opt)
        self.seq2seq.load(opt.model_list_path[1])#.to(opt.device).train()
        
        self.alpha_T_1 = nn.Linear(self.opt.encoder_hidden_size , 128)
        self.alpha_T_2 = nn.Linear(128, int((self.opt.T + self.opt.future)/2))
        
    
    def forward(self, input_data, target_data):
#        oldpath = self.seq2seq.save(self.opt)
#        print(oldpath)
#        self.seq2seq.load(self.opt.model_list_path[1])
        output_data_ss, seq2seq_alpha = self.seq2seq(input_data)
        output_data_ss, seq2seq_alpha = output_data_ss.detach(), seq2seq_alpha.detach()
        multi_N = 2 * t.sigmoid(self.alpha_T_2(t.tanh(self.alpha_T_1(seq2seq_alpha.permute(1,0,2).reshape(input_data.size(1),-1))))) # batch * 
        multi_N = t.cat([multi_N, t.from_numpy(np.flip(np.fliplr(multi_N.detach().cpu().numpy()), axis = 0).copy()).to(self.opt.device)], dim = 1)
        fft_i_t_data = t.rfft(t.cat([input_data, target_data], dim=0).squeeze(2).permute(1, 0), 2, onesided = False)
        orginal_fft_i_t_data = fft_i_t_data.detach()
        
        fft_i_t_data = fft_i_t_data * multi_N.unsqueeze(2).repeat(1,1,2)
        ifft_i_t_data = t.irfft(fft_i_t_data, 2, onesided = False).unsqueeze(0).permute(2, 1, 0)
#        self.seq2seq.load(self.opt.model_list_path[1])
        out, alpha = self.seq2seq(ifft_i_t_data[0:self.opt.T,:,:])
        
        self.drawTest(input_data.detach().cpu(), ifft_i_t_data[:self.opt.T,:,:].detach().cpu())
###
        self.drawTestF(t.norm(orginal_fft_i_t_data.detach().squeeze(0).cpu(), 2, 1).unsqueeze(1).unsqueeze(2)/self.opt.T, 
                      t.norm(fft_i_t_data.detach().squeeze(0).cpu(), 2, 1).unsqueeze(1).unsqueeze(2)/self.opt.T,
                      multi_N.detach().cpu().permute(1, 0).unsqueeze(2), ylim = 1.5) 
    
        return out, ifft_i_t_data[:self.opt.T,:,:], ifft_i_t_data[self.opt.T:,:,:]
    
    def drawTest(self, input_data, input_data_):
        batch = 0
        rows = 2
        columns = math.ceil(input_data.shape[2] * 1.0 / rows)
        n = input_data[:,batch,0].shape[0]
        for i in range(input_data.shape[2]):
            plt.subplot(rows, columns, i+1)
            print(np.arange(n).shape)
            print(input_data[:, batch, i].shape)
            plt.plot(np.arange(n), input_data[:, batch, i].detach().numpy())
            plt.plot(np.arange(n), input_data_[:, batch, i].detach().numpy())
            plt.grid()
#            plt.title('sequence')
            plt.legend(['raw data','reconstructed data'])
            plt.xlabel('time')
            plt.ylabel('amplitude')
        import time
        plt.savefig('savedfig\T_'+str(time.time())+'.svg')
        plt.show()
    
    def drawTestF(self, input_data, input_data_, multi_N, ylim = 4):
        input_data = t.clamp(input_data, 0, 1.47)
        input_data_ = t.clamp(input_data_, 0, 1.47)
        batch = 0
        rows = 2
        columns = math.ceil(input_data.shape[2] * 1.0 / rows)
        n = input_data[:,batch,0].shape[0]
        for i in range(input_data.shape[2]):
            plt.subplot(rows, columns, i+1)
            print(np.arange(n).shape)
            print(input_data[:, batch, i].shape)
            plt.plot(np.arange(n), input_data[:, batch, i].detach().numpy())
            plt.plot(np.arange(n), input_data_[:, batch, i].detach().numpy())
            plt.plot(np.arange(n), multi_N[:, batch, i].detach().numpy())
            plt.ylim([0,ylim])
            plt.grid()
#            plt.title('Frequence')
            plt.legend(['raw frequency','reconstructed frequency', 'coefficient'])
            plt.xlabel('frequency')
            plt.ylabel('magnitude')
        import time
        plt.savefig('savedfig\T_fff'+str(time.time())+'.svg')
        plt.show()

