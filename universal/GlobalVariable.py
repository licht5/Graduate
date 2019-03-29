# -*- coding: utf-8 -*-
"""
@file: GlobalVariable.py
@author: tianfeihan
@time: 2019-03-22  15:44:11
@description: 设置全局变量
"""
# rate_x, rate_y = 1,5
# MINORITY_RATIO=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,
#                 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
MINORITY_RATIO=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,
                0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
MINORITY_num=[32,3916,126,145,445,332,134,333,1017,1813,7841,23084]
project_num=0
minority=0
stop_Flag=False
test_rate = 0.2
project=["HT","MR","IS","LD","MM","TTT","CAR","CMC","MS","SB","AL","C4"]
project_name=project[project_num]
att_type="num"
att=["b","b","a","a","a","a","a","a","a","a","a","a"]
filename="../Data/rawData/"+project_name+".csv"
savename="../Data/interData/ceshi.csv"
train_filename = "../Data/interData/train.csv"
test_filename = "../Data/interData/test.csv"

excel_filename="../Data/excel/"+project_name+".xls"
alg_name=["LR","KNN","SVM","GNB","DT","MLP","AdaBT","GBDT","RF"]
alg_num=len(alg_name)
evaluation=["acc","pre","rec","f1","auc"]
evaluation_num=len(evaluation)
flag=False
count=1
# temp_excel_filename="../dataFile/tem.xls"
algrithm=[[],[],[],[],[],[],[],[],[]]

