# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:25:23 2018

@author: Ljx
"""

from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable

class Test(BasicModule):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input_data, future = 0):
        outputs = []
        h_t = t.zeros(input_data.size(0), 51, dtype=t.double)
        c_t = t.zeros(input_data.size(0), 51, dtype=t.double)
        h_t2 = t.zeros(input_data.size(0), 51, dtype=t.double)
        c_t2 = t.zeros(input_data.size(0), 51, dtype=t.double)

        for i, input_t in enumerate(input_data.chunk(input_data.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = t.stack(outputs, 1).squeeze(2)
        return outputs