# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:29:22 2018

@author: Ljx
"""
import warnings

from .ExchangeRate import ExchangeRate
from .NINO import NINO
from .Yahoo import Yahoo
from .Wecar import Wecar
from .Aircraft import Aircraft
from .BJpm import BJpm
from .GEFCom2014_Task1_P import GEFCom2014_Task1_P

class datasets(object):
    def __init__(self, opt, for_test_data = False):
        self.o = opt
        self.for_test_data = for_test_data
        
    def getData(self):
        if self.o.data == 'ExchangeRate':
            return ExchangeRate(self.o)
        if self.o.data == 'NINO':
            return NINO(self.o, self.for_test_data)
        if self.o.data == 'Yahoo':
            return Yahoo(self.o)
        if self.o.data == 'Wecar':
            return Wecar(self.o)
        if self.o.data == 'Aircraft':
            return Aircraft(self.o)
        if self.o.data == 'BJpm':
            return BJpm(self.o)
        if self.o.data == 'GEFCom2014_Task1_P':
            return GEFCom2014_Task1_P(self.o)
        warnings.warn('Warning: opt has not attribute %s' % self.opt.data)
    