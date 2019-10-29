# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:07:23 2019

@author: Ljx
"""
from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class Seq2Seq(BasicModule):
    def __init__(self, opt):      
        super(Seq2Seq, self).__init__()
        self.module_name = 'Seq2Seq'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        
        self.encoder = nn.LSTM(self.input_size, self.encoder_hidden_size, 1)
        self.encoder_decoder_h = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.encoder_decoder_c = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.attention = nn.Sequential(nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size,
                                                 self.encoder_hidden_size),
                                       nn.Tanh(),
                                       nn.Linear(self.encoder_hidden_size, 1))
        self.decoder = nn.LSTMCell(self.encoder_hidden_size, self.decoder_hidden_size)
        self.decoder_in = nn.Linear(self.input_size, self.output_size)
        self.out_linear = nn.Linear(self.output_size + self.decoder_hidden_size, self.output_size)
    
    def forward(self, input_data, r_len = 1):
        encoder_hidden = self.init_encoder_inner(input_data)
        encoder_cell = self.init_encoder_inner(input_data)
        en_outs_h, (en_h_out, en_c_out) = self.encoder(input_data, (encoder_hidden, encoder_cell))
        decoder_hidden = self.encoder_decoder_h(en_h_out[0])
        decoder_cell = self.encoder_decoder_c(en_c_out[0])
        out_data = t.zeros(self.opt.future, en_outs_h.size(1), self.output_size, dtype=t.float64).to(self.opt.device)
        out_data_temp = self.decoder_in(input_data[-1])
        alpha = t.zeros(self.opt.future, en_outs_h.size(1), int(self.opt.T/r_len)).to(self.opt.device)
        for i in range(self.opt.future):
            # batch_size * T * (2 * decoder_hidden_size + encoder_hidden_size)
            temp = t.cat((decoder_hidden.unsqueeze(0).repeat(int(self.opt.T/r_len), 1, 1).permute(1, 0, 2),
                          decoder_cell.unsqueeze(0).repeat(int(self.opt.T/r_len), 1, 1).permute(1, 0, 2),
                          en_outs_h.permute(1, 0, 2)),dim = 2)
            # batch_size * T
            temp = nn.functional.softmax(self.attention( \
                    temp.reshape(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size \
                              )).reshape(-1, int(self.opt.T/r_len)), dim = 1)
            alpha[i,:,:] = temp
            # context_attention : batch_size * encoder_hidden_size
            context_attention = t.bmm(temp.unsqueeze(1), en_outs_h.permute(1, 0, 2))[:,0,:]
            decoder_hidden, decoder_cell = self.decoder(context_attention, (decoder_hidden, decoder_cell))
            
            out_data_temp = self.out_linear(t.cat((out_data_temp, decoder_hidden), dim = 1))
            out_data[i, :, :] = out_data_temp
        return out_data, en_h_out
        
    def init_encoder_inner(self, x):
        return Variable(x.data.new(1, x.size(1), self.encoder_hidden_size).zero_())
    