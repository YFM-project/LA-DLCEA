"""
    LA-DLCEA
"""
from readDataSet import *
from Initial import *
from CalfitnessEEDHFHFLSP import *
import os
import numpy as np
from GA import *
import torch
import math
from Loperators import *
from Dueling_DDQN1 import *
from Dueling_DDQN2 import *
from Dueling_DDQN3 import *
from Energy_Saving import *
import time
np.set_printoptions(threshold=math.inf)

# 设置数据集
filename = []
filenamers = []
filevelname = []

Job = [20, 40, 60, 80, 100, 120]
Factory = [2, 3, 4, 5, 6, 7]
Stage = [6, 8, 10, 12]

for i in range(len(Job)):
    for j in range(len(Factory)):
        for k in range(len(Stage)):
            filename.append(f'DataSet/dataset_{Job[i]}J_{Factory[j]}F_{Stage[k]}S.txt')
            filenamers.append(f'dataset_{Job[i]}J_{Factory[j]}F_{Stage[k]}S')
            filevelname.append(f'DataSet/dataset_{Job[i]}J_{Factory[j]}F_{Stage[k]}S_vel.txt')
# 种群
popsize = 100
Pc=1.0
Pm=0.3
# 神经网络参数
lr = 0.001                  # learning rate
batch_size = 16             # 一批大小
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency
MEMORY_CAPACITY = 512
EPOCH = 1
N_ACTION1 = 2
N_ACTION2 = 4
N_ACTION3 = 4

for file in range(len(filename)):
    print(filename[file])
    # 读取数据集
    N, F, S, MS, processTime, lot_streaming, vel = readDataSet(filename[file], filevelname[file])
    # 评估次数
    MaxNFEs = 100 * N
    # 创建文件路径来存储每次独立运行的pareto解集
    respath = 'Result\\LA-DLCEA\\'
    sprit = '\\'
    respath = respath + filenamers[file]
    isExist = os.path.exists(respath)
    # if the result path has not been created
    if not isExist:
        currentpath = os.getcwd()
        os.makedirs(currentpath + sprit + respath)
    print(filenamers[file], 'is being Optimizing\n')
    for rround in range(0, 5):
        startTime = time.time()
        # 初始化
        p_chrom, f_chrom = CHinitial(popsize, N, F, S, MS, processTime, lot_streaming, vel)
        # 评价初始种群
        fitness = np.zeros(shape=(popsize, 3))
        for i in range(popsize):
            fitness[i, 0], fitness[i, 1], fitness[i, 2] = FitEEDHFHFLSP(p_chrom[i, :], f_chrom[i, :], N, F, S, MS, processTime, lot_streaming, vel)
        # 函数评估次数
        NFEs = 0
        it = 1
        # 精英档案集
        AP_chrom = []
        AF_chrom = []
        AFitness = []
        # Dueling_DDQN初始化
        N_STATE1 = (3 * N) + 2
        N_STATE2 = N_STATE1 + 2
        N_STATE3 = N_STATE1 + 2
        ddqn_net1 = DuelingDQN1(N_STATE1, N_ACTION1,BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA = GAMMA, MEMORY_CAPACITY= MEMORY_CAPACITY, TARGET_REPLACE_ITER = TARGET_REPLACE_ITER)
        ddqn_net2 = DuelingDQN2(N_STATE2, N_ACTION2,BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA = GAMMA, MEMORY_CAPACITY= MEMORY_CAPACITY, TARGET_REPLACE_ITER = TARGET_REPLACE_ITER)
        ddqn_net3 = DuelingDQN3(N_STATE3, N_ACTION3, BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA=GAMMA, MEMORY_CAPACITY=MEMORY_CAPACITY, TARGET_REPLACE_ITER=TARGET_REPLACE_ITER)
        # 算法迭代开始
        while NFEs < MaxNFEs:
            print(filename[file] + ' ' + 'rround', rround + 1, 'iter', it)
            # 初始化局部搜索状态
            current_state1 = np.zeros(N_STATE1, dtype=int)
            next_state1 = np.zeros(N_STATE1, dtype=int)
            current_state2 = np.zeros(N_STATE2, dtype=int)
            next_state2 = np.zeros(N_STATE2, dtype=int)
            current_state3 = np.zeros(N_STATE3, dtype=int)
            next_state3 = np.zeros(N_STATE3, dtype=int)
            Fit = np.zeros(3)
            # 全局搜索
            if it < math.ceil(((MaxNFEs * 2) / (popsize * 6))):
                p_chrom, f_chrom, fitness = NSGA2(p_chrom, f_chrom, fitness, Pc, Pm, popsize, N, F, S, MS, processTime, lot_streaming, vel)
                NFEs = NFEs + 2 * popsize
            else:
                p_chrom, f_chrom, fitness = INSGA2(p_chrom, f_chrom, fitness, Pc, Pm, popsize, N, F, S, MS, processTime, lot_streaming, vel)
                NFEs = NFEs + 2 * popsize
            it = it + 1
            # 局部搜索
            # 获得非支配解
            # 精英策略
            PF = pareto(fitness)
            if len(AFitness) == 0:
                AP_chrom = copy.copy(p_chrom[PF, :])
                AF_chrom = copy.copy(f_chrom[PF, :])
                AFitness = copy.copy(fitness[PF, :])
            else:
                AP_chrom = np.vstack((AP_chrom, p_chrom[PF, :]))
                AF_chrom = np.vstack((AF_chrom, f_chrom[PF, :]))
                AFitness = np.vstack((AFitness, fitness[PF, :]))
            PF = pareto(AFitness)
            AP_chrom = AP_chrom[PF, :]
            AF_chrom = AF_chrom[PF, :]
            AFitness = AFitness[PF, :]
            AP_chrom, AF_chrom, AFitness = DeleteReaptE(AP_chrom, AF_chrom, AFitness)
            # count = len(PF)
            # 局部搜索
            for pop in range(len(AFitness)):
                # 选择优化Cmax还是TEC
                current_state1[0:N] = copy.copy(AP_chrom[pop, :]) # JS序列
                current_state1[N:(N * 2)] = copy.copy(AF_chrom[pop, :]) # FA序列
                for lot in range(N):
                    current_state1[(N*2) + lot] = lot_streaming[AP_chrom[pop, lot]] # 批次大小
                current_state1[(N * 3):(N * 3) + 2] = copy.copy(AFitness[pop, 0: 2]) # 两个目标值（模糊完工时间和模糊加工能耗）
                # TEC+Cmax
                aCmaxTEC = ddqn_net1.return_probs(current_state1)
                # select action
                action1 = ddqn_net1.choose_action(current_state1)
                a1 = int(action1)
                if a1 == 0:
                    # 选择更新以Cmax为验收标准的IG算子
                    current_state2[0:(N*3)+2] = copy.copy(current_state1[0:(N*3)+2])
                    current_state2[(N*3)+2:(N*3)+2+2] = copy.copy(aCmaxTEC[0:2])
                    action2 = ddqn_net2.choose_action(current_state2)
                    a2 = int(action2)
                    # 选择算子
                    if a2 == 0:
                        P1, F1 = SER_IG1_Cmax(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a2 == 1:
                        P1, F1 = SER_IG2_Cmax(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a2 == 2:
                        P1, F1 = SER_IG3_Cmax(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a2 == 3:
                        P1, F1 = SER_IG4_Cmax(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    Fit[0], Fit[1], Fit[2] = FitEEDHFHFLSP(P1, F1, N, F, S, MS, processTime, lot_streaming, vel)
                    NFEs = NFEs + 1
                    flag2 = NDS(Fit, AFitness[pop, :])
                    if flag2 == 1:
                        AP_chrom[pop, :] = copy.copy(P1)
                        AF_chrom[pop, :] = copy.copy(F1)
                        AFitness[pop, :] = copy.copy(Fit)
                        reward2 = 20 + ((0.25*(Fit[0] - AFitness[pop, 0]))+(0.75*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    elif flag2 == 0:
                        AP_chrom = np.vstack((AP_chrom, P1))
                        AF_chrom = np.vstack((AF_chrom, F1))
                        AFitness = np.vstack((AFitness, Fit))
                        if Fit[0] < AFitness[pop, 0]:
                            reward2 = 15+ ((0.25*(Fit[0] - AFitness[pop, 0]))+(0.75*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                        else:
                            reward2 = 0+ ((0.25*(Fit[0] - AFitness[pop, 0]))+(0.75*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                        if Fit[1] < AFitness[pop, 1]:
                            reward2 = 10+ ((0.25*(Fit[0] - AFitness[pop, 0]))+(0.75*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    else:
                        reward2 = 0+ ((0.25*(Fit[0] - AFitness[pop, 0]))+(0.75*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    next_state2[0:N] = copy.copy(P1)
                    next_state2[N:(N * 2)] = copy.copy(F1)
                    for lot1 in range(N):
                        next_state2[(N * 2) + lot1] = lot_streaming[P1[lot1]]  # 批次大小
                    next_state2[(N * 3):(N * 3) + 2] = copy.copy(Fit[0:2])
                    next_state2[(N * 3) + 2:(N * 3) + 2 + 2] = copy.copy(aCmaxTEC[0:2])
                    ddqn_net2.store_transition(current_state2, action2, reward2, next_state2)
                    if ddqn_net2.memory_counter > 50:
                        for epoch2 in range(EPOCH):
                            ddqn_net2.learn()
                elif a1 == 1:
                    # 选择更新以TEC为验收标准的IG算子
                    current_state3[0:(N * 3) + 2] = copy.copy(current_state1[0:(N * 3) + 2])
                    current_state3[(N * 3) + 2:(N * 3) + 2 + 2] = copy.copy(aCmaxTEC[0:2])
                    action3 = ddqn_net3.choose_action(current_state3)
                    a3 = int(action3)
                    if a3 == 0:
                        P1, F1 = SER_IG1_TEC(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a3 == 1:
                        P1, F1 = SER_IG2_TEC(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a3 == 2:
                        P1, F1 = SER_IG3_TEC(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    elif a3 == 3:
                        P1, F1 = SER_IG4_TEC(AP_chrom[pop, :], AF_chrom[pop, :], int(AFitness[pop, 2]), N, F, S, MS, processTime, lot_streaming, vel)
                    Fit[0], Fit[1], Fit[2] = FitEEDHFHFLSP(P1, F1, N, F, S, MS, processTime, lot_streaming, vel)
                    NFEs = NFEs + 1
                    flag3 = NDS(Fit, AFitness[pop, :])
                    if flag3 == 1:
                        AP_chrom[pop, :] = copy.copy(P1)
                        AF_chrom[pop, :] = copy.copy(F1)
                        AFitness[pop, :] = copy.copy(Fit)
                        reward3 = 20 + ((0.75*(Fit[0] - AFitness[pop, 0]))+(0.25*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    elif flag3 == 0:
                        AP_chrom = np.vstack((AP_chrom, P1))
                        AF_chrom = np.vstack((AF_chrom, F1))
                        AFitness = np.vstack((AFitness, Fit))
                        if Fit[0] < AFitness[pop, 0]:
                            reward3 = 15 + ((0.75*(Fit[0] - AFitness[pop, 0]))+(0.25*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                        else:
                            reward3 = 0 + ((0.75*(Fit[0] - AFitness[pop, 0]))+(0.25*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                        if Fit[1] < AFitness[pop, 1]:
                            reward3 = 10 + ((0.75*(Fit[0] - AFitness[pop, 0]))+(0.25*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    else:
                        reward3 = 0 + ((0.75*(Fit[0] - AFitness[pop, 0]))+(0.25*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    next_state3[0:N] = copy.copy(P1)
                    next_state3[N:(N * 2)] = copy.copy(F1)
                    for lot2 in range(N):
                        next_state3[(N * 2) + lot2] = lot_streaming[P1[lot2]]  # 批次大小
                    next_state3[(N * 3):(N * 3) + 2] = copy.copy(Fit[0:2])
                    next_state3[(N * 3) + 2:(N * 3) + 2 + 2] = copy.copy(aCmaxTEC[0:2])
                    ddqn_net3.store_transition(current_state3, action3, reward3, next_state3)
                    if ddqn_net3.memory_counter > 50:
                        for epoch3 in range(EPOCH):
                            ddqn_net3.learn()
                # 外围Dueling_DDQN
                flag1 = NDS(Fit, AFitness[pop, :])
                if flag1 == 1:
                    reward1 = 20 + ((0.5*(Fit[0] - AFitness[pop, 0]))+(0.5*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                elif flag1 == 0:
                    if Fit[0] < AFitness[pop, 0]:
                        reward1 = 15 + ((0.5*(Fit[0] - AFitness[pop, 0]))+(0.5*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    else:
                        reward1 = 0 + ((0.5*(Fit[0] - AFitness[pop, 0]))+(0.5*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                    if Fit[1] < AFitness[pop, 1]:
                        reward1 = 10 + ((0.5*(Fit[0] - AFitness[pop, 0]))+(0.5*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                else:
                    reward1 = 0 + ((0.5*(Fit[0] - AFitness[pop, 0]))+(0.5*(Fit[1] - AFitness[pop, 1])))/(N*1000)
                next_state1[0:N] = copy.copy(P1)
                next_state1[N:(N * 2)] = copy.copy(F1)
                for lot3 in range(N):
                    next_state1[(N * 2) + lot3] = lot_streaming[P1[lot3]]  # 批次大小
                next_state1[(N * 3):(N * 3) + 2] = copy.copy(Fit[0:2])
                ddqn_net1.store_transition(current_state1, action1, reward1, next_state1)
                if ddqn_net1.memory_counter > 50:
                    for epoch1 in range(EPOCH):
                        ddqn_net1.learn()
        # 节能策略
        Length = len(AFitness)
        for length in range(Length):
            AFitness[length, 0], AFitness[length, 1], AFitness[length, 2] = Energy_FitEEDHFHFLSP(AP_chrom[length, :], AF_chrom[length, :], N, F, S, MS, processTime, lot_streaming, vel)