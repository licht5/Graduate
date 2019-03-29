# -*- coding: utf-8 -*-
"""
@file: SetDataUnbalanced.py
@author: tianfeihan
@time: 2019-03-22  16:11:03
@description: 
"""
from sklearn.preprocessing import Imputer
import numpy as np
import csv
import random
from universal import GlobalVariable as gv
from sklearn.model_selection import train_test_split
def SetDataUnbalancedFunc(filename,test_rate,att_add,minority,minority_num_a):
    data = np.loadtxt(filename,  delimiter=',')
    if att_add == "a":
        Attributes, targets = data[:, :-1], data[:, -1]
    else:
        Attributes, targets = data[:, 1:], data[:, 0]
    tatol_num=len(targets)
    minority_num=minority_num_a
    majority_num=tatol_num-minority_num

    test_num=int(test_rate*tatol_num)
    test_num=min(test_num,1500)
    test_num_minority=int(test_num*minority_num/tatol_num)
    test_num_majority=test_num-test_num_minority

    minority_left_num=minority_num-test_num_minority
    majority_left_num=majority_num-test_num_majority

    s1=int(majority_left_num/(1-gv.MINORITY_RATIO[0]))
    s2=int(minority_left_num/gv.MINORITY_RATIO[-1])
    # s1=int(majority_left_num/0.8)
    # s2=int(minority_left_num/0.8)
    # print("s1:"+str(s1)+"s2:"+str(s2)+"minority_left_num:"+str(minority_left_num))


    train_num=min(s1,s2,3500)
    train_num_minority=int(train_num*minority)
    train_num_majority=train_num-train_num_minority

    # print(train_num_minority,train_num_majority,test_num_minority,test_num_majority)


    train_min_count=0
    train_maj_count = 0
    test_min_count = 0
    test_maj_count = 0

    train_data = []
    test_data = []

    for i in range(len(targets)):
        random_num = random.randint(0, len(data) - 1)
        data_tem = data[random_num]
        np.delete(data, random_num, axis=0)
        if att_add=="a":
            att=data_tem[-1]
        else:
            att=data_tem[0]
        if int(att) == 1:
            if test_min_count<test_num_minority:
                test_min_count=test_min_count+1
                test_data.append(data_tem)
            elif train_min_count<train_num_minority:
                train_min_count=train_min_count+1
                train_data.append(data_tem)
            else:
                pass

        elif int(att)==0:
            if test_maj_count<test_num_majority:
                test_maj_count=test_maj_count+1
                test_data.append(data_tem)
            elif train_maj_count<train_num_majority:
                train_data.append(data_tem)
                train_maj_count=train_maj_count+1
            else:
                pass
        else:
            print("============= class error===========")
        if train_min_count+train_maj_count>=train_num:
            break
    np.savetxt("../Data/interData/train.csv", train_data, delimiter=',')
    np.savetxt("../Data/interData/test.csv", test_data, delimiter=',')

# if __name__ == '__main__':
#     test_rate=0.01
#     filename="../Data/rawData/CMC.csv"
#     SetDataUnbalancedFunc(filename,test_rate,"a",0.8,333)