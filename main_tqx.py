#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import mainAE
import mainSS
import mainR
import mainF
from config import opt

run_data = 'NINO' #NINO,ExchangeRate
opt.data_list_index = 1

ttttrain = False

if ttttrain:
#    aepath = mainAE.main_run(opt, run_data)
#    opt.model_list_path[0] = 'checkpoints/AutoEncoder_ExchangeRate_0124123132.pth'
    
    sspath = mainSS.main_run(opt, run_data)
    opt.model_list_path[1] = sspath
    
#    rpath = mainR.main_run(opt, run_data)
#    opt.model_list_path[2] = rpath
    
#    fpath = mainF.main_run(opt, run_data)
#    opt.model_list_path[3] = fpath

    opt._parse(printconfig = True)

if not ttttrain:
    opt.train_test_pre = 2
#    opt.model_list_path = ['checkpoints/AutoEncoder_ExchangeRate_0123205823.pth', 'checkpoints/Seq2Seq_NINO_0123193337.pth', 'checkpoints/Recurrent_NINO_0123024345.pth', 'checkpoints/multiF_NINO_0123061425.pth']
    opt.model_list_path = ['checkpoints/AutoEncoder_NINO_0123162251.pth', 'checkpoints/Seq2Seq_NINO_0123193230.pth', 'checkpoints/Recurrent_NINO_0123232459.pth', 'checkpoints/multiF_NINO_0124042149.pth']
#    aepath = mainAE.main_run_pre(opt, run_data)
#    sspath = mainSS.main_run_pre(opt, run_data)
    rpath = mainR.main_run_pre(opt, run_data)
#    fpath = mainF.main_run_pre(opt, run_data)






