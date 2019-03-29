# -*- coding: utf-8 -*-
"""
@file: algorithm.py
@author: tianfeihan
@time: 2019-03-22  16:10:03
@description: 
"""
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xlwt,xlrd

from sklearn.metrics import classification_report
from sklearn import metrics, preprocessing
from sklearn import tree
from sklearn import svm

import os
from xlutils.copy import copy
import universal.GlobalVariable   as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import sklearn.model_selection
def getData(att):
    train_filename = gv.train_filename
    test_filename = gv.test_filename
    train_data_total=np.loadtxt(train_filename,delimiter=",")
    test_data_total = np.loadtxt(test_filename, delimiter=",")
    if(att=="a"):
        train_data, train_target=train_data_total[:,:-1],train_data_total[:,-1]
        ts_data, ts_target = test_data_total[:,:-1],test_data_total[:,-1]
    else:
        train_data, train_target = train_data_total[:, 1:], train_data_total[:, 0]
        ts_data, ts_target = test_data_total[:, 1:], test_data_total[:, 0]
    if not gv.flag:
        shuju_1 = sum(train_target)
        print("少数类的训练样本:" + str(shuju_1))
        print("多数类的训练样本:" + str(len(train_target) - shuju_1))

        shuju_2 = sum(ts_target)
        print("少数类的测试样本:" + str(shuju_2))
        print("多数类的测试样本:" + str(len(ts_target) - shuju_2))
        gv.flag=True


    return train_data,ts_data,train_target,ts_target





# ******************************* execel *******************************
# excel_filename=gv.excel_filename
alg_name=gv.alg_name


# ******************************* 显示评价指标 *******************************
def showResult(result,i,ts_target):
    acc= metrics.accuracy_score(ts_target, result)
    pre= metrics.precision_score(ts_target, result)
    rec=metrics.recall_score(ts_target, result)
    f1= metrics.f1_score(ts_target, result)
    auc=metrics.roc_auc_score(ts_target, result)
    tem = []
    tem.append(acc)
    tem.append(pre)
    tem.append(rec)
    tem.append(f1)
    tem.append(auc)
    gv.algrithm[i].append(tem)

# ******************************* 逻辑斯特回归 *******************************
def LogistRe(train_data,ts_data, train_target,ts_target):
    clf = LogisticRegression(random_state=0)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result, 0, ts_target)


def KNNClass(train_data,ts_data, train_target,ts_target):
    clf = KNeighborsClassifier(n_neighbors=2)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result, 1, ts_target)



# *******************************支持向量机 70/345*******************************
def svcFunc(train_data,ts_data, train_target,ts_target):
    clf=svm.SVC(probability=True)
    clf=clf.fit(train_data,train_target)
    result=clf.predict(ts_data)
    showResult(result,2,ts_target)


#*******************************高斯朴素贝叶斯 77/345*******************************
def gsNB(train_data,ts_data, train_target,ts_target):
    gnb = GaussianNB()
    clf = gnb.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,3,ts_target)



#******************************* 决策树*******************************
def deTree(train_data,ts_data, train_target,ts_target):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,4,ts_target)

#******************************* 神经网络  *******************************
def MLPClass(train_data,ts_data, train_target,ts_target):
    clf = MLPClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,5,ts_target)

#******************************* adaboost 的集成方法 *******************************
def adaBoost(train_data,ts_data, train_target,ts_target):
    clf = AdaBoostClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,6,ts_target)

#******************************* GBDT 的集成方法 *******************************
def GBDT(train_data,ts_data, train_target,ts_target):
    clf = GradientBoostingClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    showResult(result,7,ts_target)

# *******************************随机森林的集成方法  n_estimators为5时，准确率最高 *******************************
def rdForest(train_data,ts_data, train_target,ts_target):
    clf = RandomForestClassifier()
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result,8,ts_target)

def  totalAlgrithon(att):
    print("==================="+str(gv.count)+"次算法 =====================")
    gv.count=gv.count+1
    train_data,ts_data, train_target,ts_target=getData(att)
    train_data = preprocessing.scale(train_data)
    ts_data=preprocessing.scale(ts_data)
    LogistRe(train_data, ts_data, train_target, ts_target)
    KNNClass(train_data, ts_data, train_target, ts_target)
    svcFunc(train_data,ts_data, train_target,ts_target)
    gsNB(train_data,ts_data, train_target,ts_target)
    deTree(train_data,ts_data, train_target,ts_target)
    MLPClass(train_data, ts_data, train_target, ts_target)
    adaBoost(train_data,ts_data, train_target,ts_target)
    GBDT(train_data, ts_data, train_target, ts_target)
    rdForest(train_data,ts_data, train_target,ts_target)