# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:25:56 2018

@author: Ljx
"""

from .BasicModule import BasicModule

import torch as t, numpy as np
import torch.nn as nn
from .kpywavelet import wavelet as kpywavelet

from torch.autograd import Variable

class Wavelet_ATT(BasicModule):
    
    def __init__(self, opt):
        super(Wavelet_ATT, self).__init__()
        self.module_name = 'Wavelet_ATT'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        self.lstm_layer = 2
        self.cnn_out_channel = 32
        
        self.encoder = nn.LSTM(self.input_size, self.encoder_hidden_size, self.lstm_layer)
        self.decoder_in = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.attention = nn.Sequential(nn.Linear(self.encoder_hidden_size + self.cnn_out_channel,
                                                 self.encoder_hidden_size),
                                       nn.Tanh(),
                                       nn.Linear(self.encoder_hidden_size, 1))
        
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=self.input_size, out_channels=self.input_size*2, kernel_size=(3,3),bias = False),
                nn.BatchNorm2d(self.input_size*2),
                nn.Conv2d(in_channels=self.input_size*2, out_channels=self.input_size*4, kernel_size=(3,3), bias = False),
                nn.BatchNorm2d(self.input_size*4),
                nn.Conv2d(in_channels=self.input_size*4, out_channels=self.cnn_out_channel, kernel_size=(3,3), bias = False),
                )
        self.together = nn.Linear(self.opt.T * self.encoder_hidden_size + (self.opt.T-6)*self.cnn_out_channel,
                                  self.decoder_hidden_size * opt.future)
        self.out_linear = nn.Linear(self.decoder_hidden_size * opt.future, self.opt.future * self.output_size)

    # input_data : T * batch_size * 1(input_size) 
    def forward(self, input_data):
        encoder_hidden = self.init_encoder_inner(input_data)
        encoder_cell = self.init_encoder_inner(input_data)
        en_outs_h, (en_h_out, en_c_out) = self.encoder(input_data, (encoder_hidden, encoder_cell))
        
        wavelet_power = self.WaveletTransform(input_data)
        cnn_out = self.cnn(wavelet_power)
        
        # cnn_out_att : cnn_T * batch * cnn_out_channel
        cnn_out_att = t.zeros(cnn_out.size(3), cnn_out.size(0), cnn_out.size(1)).to(self.opt.device)
        for i in range(6, self.opt.T):
            temp = t.cat((en_outs_h[i].unsqueeze(0).repeat(cnn_out.size(2), 1, 1).permute(1,0,2),
                          cnn_out[:,:,:,i-6].permute(0, 2, 1)), dim = 2) # batch * f * size
            temp = nn.functional.softmax(self.attention(
                    temp.reshape(-1, self.encoder_hidden_size + self.cnn_out_channel)).reshape(-1, cnn_out.size(2)),dim = 1)
            context_attention = t.bmm(temp.unsqueeze(1), cnn_out[:,:,:,i-6].permute(0,2,1))[:,0,:]
            cnn_out_att[i-6,:,:] = context_attention
        together_out = self.together(t.cat((en_outs_h.permute(1,0,2).reshape(input_data.size(1),-1),
                                            cnn_out_att.permute(1,0,2).reshape(input_data.size(1),-1)), dim =1))
        out_data = self.out_linear(nn.functional.relu(together_out)).reshape(input_data.size(1),self.opt.future,self.output_size)
        out_data = out_data.permute(1, 0, 2)
        
        return out_data
        
        
    def init_encoder_inner(self, x):
        return Variable(x.data.new(self.lstm_layer, x.size(1), self.encoder_hidden_size).zero_())
        
        
    def WaveletTransform(self, input_data):
#        data = (input_data - input_data.mean(dim = 0)) / input_data.std(dim = 0) for exchange data
        data = input_data
        dt = 1
        dj = 1./20
        mother = kpywavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6
        # data_ : batch * size     *      T
        data_ = data.reshape(data.size(0), data.size(1) * data.size(2)).permute(1,0).cpu().numpy()
#        print('f',data_[28, ...])
        f_dim = int(np.log2(data.size(0) / 2) / dj) + 1
        t_dim = int(data.size(0))
        power = np.zeros([data.size(1) * data.size(2), f_dim, t_dim])
        for i in range(data_.shape[0]):
#            print(data_.shape, dt, dj)
#            print(i,data_[i, ...])
#            print(i)
            wave, scales, freqs, coi, dj, s0, J = kpywavelet.cwt(data_[i, ...], dt, dj=dj, s0=-1, J=-1, wavelet=mother)
#            print(wave.shape)
#            print(power[0].shape)
            power[i] = np.power(wave, 2)
        power = t.from_numpy(power).reshape(data.size(1), data.size(2), f_dim, t_dim).to(self.opt.device)
        #power : batch * input_data_size * F.dim * T.dim
        return power
        
        