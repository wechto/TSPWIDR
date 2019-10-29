#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import mainAE
import mainSS
import mainR
import mainF

import mainN

from config import opt

run_data = 'NINO'
opt.data_list_index = 3

''' 

ttttrain = False

if ttttrain:
    
#    aepath = mainAE.main_run(opt, run_data)
    opt.model_list_path[0] = 'checkpoints/AutoEncoder_NINO_0123162328.pth'
#    sspath = mainSS.main_run(opt, run_data)
    opt.model_list_path[1] = 'checkpoints/Recurrent_NINO_0206222418.pth'
    rpath = mainR.main_run(opt, run_data)
    opt.model_list_path[2] = rpath
    fpath = mainF.main_run(opt, run_data)
    opt.model_list_path[3] = fpath
    opt._parse(printconfig = True)

if not ttttrain:
    opt.train_test_pre = 0
    opt.model_list_path = ['checkpoints/AutoEncoder_NINO_0123162228.pth', 'checkpoints/Recurrent_NINO_0206222215.pth', 'checkpoints/Recurrent_NINO_0208045318.pth', 'checkpoints/multiF_NINO_0208154733.pth']

#    aepath = mainAE.main_run_pre(opt, run_data)
#    sspath = mainSS.main_run_pre(opt, run_data)
    rpath = mainR.main_run_pre(opt, run_data)
    fpath = mainF.main_run_pre(opt, run_data)
    
''' 
    
''' train
opt.recurrent_n = 2
opt.model_list_path[1] = 'checkpoints/Seq2Seq_NINO_0123193051.pth'
rnpath = mainN.main_run(opt, run_data)
print('rn path:', rnpath)
'''

''' train multi
recurrent_n_max = 3
recurrent_n_path = [None]*recurrent_n_max
opt.model_list_path[1] = 'checkpoints/Seq2Seq_NINO_0123193437.pth'
for i in range(recurrent_n_max):
    opt.recurrent_n = i+2
    rnpath = mainN.main_run(opt, run_data)
    print('recurrent_n', opt.recurrent_n, 'path', rnpath)
    recurrent_n_path[i] = rnpath

print(recurrent_n_path) 
'''

''' test 
opt.recurrent_n = 3
opt.model_list_path =[None, 'checkpoints/Seq2Seq_NINO_0123193230.pth', 'checkpoints/RecurrentN_NINO_0211195451.pth', None]
MSE, U = mainN.main_run_pre(opt, run_data)
print( MSE, U )
'''

#''' test multi
import matplotlib.pyplot as plt
import numpy as np, torch as t
import math

index = [['checkpoints/Seq2Seq_NINO_0123193051.pth',
          'checkpoints/Recurrent_NINO_0123232400.pth',
          'checkpoints/RecurrentN_NINO_0212055443.pth',
          'checkpoints/RecurrentN_NINO_0212154529.pth', 
          'checkpoints/RecurrentN_NINO_0213024720.pth'], #index 0

         ['checkpoints/Seq2Seq_NINO_0123193230.pth',
          'checkpoints/Recurrent_NINO_0123232459.pth',
          'checkpoints/RecurrentN_NINO_0212055525.pth', 
          'checkpoints/RecurrentN_NINO_0212154546.pth', 
          'checkpoints/RecurrentN_NINO_0213024707.pth'], #index 1
          
         ['checkpoints/Seq2Seq_NINO_0123193337.pth',
          'checkpoints/Recurrent_NINO_0123232535.pth',
          'checkpoints/RecurrentN_NINO_0212055803.pth', 
          'checkpoints/RecurrentN_NINO_0212154908.pth', 
          'checkpoints/RecurrentN_NINO_0213024908.pth'], #index 2
          
         ['checkpoints/Seq2Seq_NINO_0123193437.pth',
          'checkpoints/Recurrent_NINO_0123232604.pth',
          'checkpoints/RecurrentN_NINO_0212055212.pth', 
           'checkpoints/RecurrentN_NINO_0212153659.pth', 
           'checkpoints/RecurrentN_NINO_0213023524.pth']] #index 3
MSEs, Us = np.zeros([4,5]), np.zeros([4,5])
for i in range(-4):
    opt.data_list_index = i
    for j in range(5):
        opt.recurrent_n = j
        opt.model_list_path =[None, index[i][0], None if j==0 else index[i][j], None]
        MSE, U = mainN.main_run_pre(opt, run_data)
        MSEs[i,j], Us[i,j] = MSE, U
print(MSEs,Us)
#
#MSEs = np.array([[1.19935327, 0.44766346, 1.14411952, 1.17249683, 1.13329989],
# [0.7814886 , 0.28622022, 0.53806496 ,0.58601214, 1.75423543],
# [0.36341466, 0.15255875, 0.21720217, 0.23372307, 0.22571262],
# [0.20302946, 0.06328554 ,0.64642089 ,0.84699274, 0.74393498]])
             
    
MSEs = np.array([[1.1994,	0.4343,	0.3712,	0.3478,	0.3395],
 [0.7851,	0.3309,	0.3297,	0.3104,	0.2641],
 [0.3634,	0.1933,	0.1797,	0.1683,	0.1503],
 [0.203,	0.113,	0.1098,	0.1006,	0.1064]])         
             




plt.figure()
for i in range(4):
    plt.plot([0, 1,2,3,4],MSEs[i,:])
plt.legend(['ENSO 1-2','ENSO 3','ENSO 3-4','ENSO 4'])
plt.grid()

gamma = np.array([9,	0.007332864,	0.00135719,
8,	0.00875126	,0.001554874,
7,	0.006605908	,0.001126431,
6,	0.005543081	,0.001536232,
5,	0.007102284	,0.004847112,
4,	0.005554814,	0.001784069,
3,	0.005522578,	0.001364947,
2,	0.001986593,	0.000956078,
1,	0.004345652,	0.001784546,
0.5,	0.000893087,	0.003177977,
0.2,	0.003450265,	0.004654602,
0.1,	0.002527938,	0.004027253,
0.067,	0.008045462,	0.001564802,
0.05,	0.014167879,	0.003121907,
0.04,	0.005019571,	0.002034204,
0.033,	0.002786551,	0.002992637])
gamma = gamma.reshape((-1,3))

plt.figure()
for i in range(2):
    plt.plot(gamma[:,i+1])
plt.legend(['RPR','HFP'])
plt.grid()

plt.figure()
x = ['0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0.05','0.02','0.01','0.0067','0.005']
xx = ['1/9','1/8','1/7','1/6','1/5','1/4','1/3','1/2','1','2','5','10','15','20','25','30']
plt.plot(xx[:-2],gamma[:-2,1])
plt.legend(['RPR'])
plt.grid()


plt.figure()
plt.subplot(121)
line = ['-o','-d','-s','-x']
for i in range(4):
    plt.plot(['0', '1','2','3','4'],MSEs[i,:], line[i])
plt.legend(['ENSO 1-2','ENSO 3','ENSO 3-4','ENSO 4'])
plt.grid()
plt.xlabel('The recurrent number: M.')

plt.subplot(222)
xx = ['1/9','1/8','1/7','1/6',r'$\frac{1}{5}$',r'$\frac{1}{4}$',r'$\frac{1}{3}$',r'$\frac{1}{2}$','1','2','5','10','15','20','25','30']
#plt.plot(xx[4:-2],gamma[4:-2,1], color = 'g')
xx_none = [' ','  ','   ','    ','     ', '      ','       ','        ','         ','          ']
plt.plot(xx_none,gamma[4:-2,1], color = 'g')
plt.legend(['RPR'])
plt.grid()

plt.subplots_adjust(wspace=0.25, hspace=0)
plt.subplot(224)
xx = ['1/9','1/8','1/7','1/6',r'$\frac{1}{5}$',r'$\frac{1}{4}$',r'$\frac{1}{3}$',r'$\frac{1}{2}$','1','2','5','10','15','20','25','30']
plt.plot(xx[4:-2],gamma[4:-2,2], color = 'orangered')
plt.ylim([0.0001,0.014])
plt.legend(['HFP'])
plt.grid()
plt.xlabel('The ratio of $\gamma$ to $1-\gamma$.')

#plt.tight_layout()
import time
plt.savefig('savedfig\A_'+str(time.time())+'.svg')
#'''



fig = plt.figure()
plt.subplot(121)
#ax_l = fig.add_subplot(121)
line = ['-o','-d','-s','-x']
for i in range(4):
    plt.plot(['0', '1','2','3','4'],MSEs[i,:], line[i])
plt.legend(['ENSO 1-2','ENSO 3','ENSO 3-4','ENSO 4'])
plt.xlabel('M')
plt.grid()



f, axarr = plt.subplots(2, sharex=True, sharey=True)

f.suptitle('the balance factor')

axarr[0].plot(xx[4:-2],gamma[4:-2,1], color = 'g')
axarr[0].grid()
axarr[0].legend(['RPR'])

axarr[1].plot(xx[4:-2],gamma[4:-2,2], color = 'orangered')
axarr[1].grid()
axarr[1].legend(['HFP'])

plt.xlabel('the ratio of $\gamma$ to $1-\gamma$')

# 间距调整为０
f.subplots_adjust(hspace=0)
# 设置全部标签在外�?for ax in axarr:
    ax.label_outer()





