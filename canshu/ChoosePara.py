# -*- coding: utf-8 -*-
"""
@file: ChoosePara.py
@author: tianfeihan
@time: 2019-03-27  11:18:25
@description: 从待选的超参数选择随不平衡程度变化变化的参数
"""
import matplotlib
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xlwt,xlrd

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn import svm

import os
from sklearn.svm import SVC
import numpy as np
from xlutils.copy import copy
# import universal.globalVariable as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
# filename="../dataFile/mushroom.csv"

from coreCode.SetDataUnbalanced import SetDataUnbalancedFunc

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def ceping(zhanbi,att,sf):
    myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams['axes.unicode_minus'] = False

    # ============= LR ================
    tol=np.arange(5e-5,2e-4,1e-5)  #[5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,11e-5,12e-5,13e-5,14e-5,16e-5]
    C=range(1,20,1)
    intercept_scaling=np.arange(0.1,3,0.2)
    random_state=range(1,30,2)
    solver=["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    max_iter=range(50,150,5)
    multi_class=["ovr","multinomial","auto"]
    verbose=range(0,10,1)

    # ============= KNN ================
    n_neighbors=range(1,20,1)
    algorithm= ["auto",  "brute"]
    leaf_size=range(10,50,2)
    p=range(1,20,1)
    n_jobs=range(-5,5,2)

    # ============= SVM ================
    C = range(1, 20, 1)
    degree=range(1,10,1)
    # gamma=[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,5,10,100]
    gamma=np.arange(0.001,0.71,0.02)
    max_iter=range(-5,5,1)
    decision_function_shape=["ovo", "ovr"]

    # ============= nb ================
    var_smoothing=np.arange(1e-9,1e-8,1e-9)

    # ============= tree ================
    max_depth=range(1,10,1)
    min_samples_split=range(2,200,8)
    # min_samples_split = np.arange(0.001, 0.1, 0.003)
    min_samples_leaf=range(25,40,1)
    min_weight_fraction_leaf=np.arange(0,0.5,0.1)
    # max_leaf_nodes=range(2,10,1)


    # algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute']
    learning_rate=np.arange(0.01,0.2,0.01)
    # max_depth=range(1,10,1)
    # min_samples_split=range(2,30,1)
    min_samples_leaf=[0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]

    # ========== mlp ====================
    hidden_layer_sizes=[(i,) for i in range(80,120,5)]
    activation=["identity", "logistic", "tanh", "relu"]
    solver=["lbfgs","sgd","adam"]
    alpha=np.arange(0.00005,0.00014,0.00001)
    batch_size=range(100,300,50)
    learning_rate=["constant", "invscaling", "adaptive"]
    learning_rate_init=[0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.0011,0.0012,0.0013,0.0014]

    # ============= Ada ================
    n_estimators=range(30,70,2)
    # learning_rate=np.arange(0.5,2,0.1)
    algorithm=["SAMME","SAMME.R"]


    # ============= gbdt ================
    loss=["deviance", "exponential"]
    learning_rate=np.arange(0.01,0.18,0.02)
    n_estimators=range(10,130,10)
    subsample=np.arange(0.1,1.01,0.1)
    min_samples_split=range(2,10,1)
    # min_samples_leaf=range(1,10,1)
    max_depth=range(1,10,1)

    # =============== rdforest==========
    n_estimators=range(5,30,1)
    max_depth=range(1,10,1)
    min_samples_split=range(2,50,5)
    max_leaf_nodes=range(2,30,5)

    fig = plt.figure(figsize=(7,10))
    # fig = plt.figure(figsize=(14, 20))
    for j in range(3):
        filename = "../Data/interData/zhanbi" + str(zhanbi[j]) + "/train.csv"
        data = np.loadtxt(filename, delimiter=",")
        project_X = data[:, :-1]
        project_y = data[:, -1]
        shuju_1 = sum(project_y)
        print("少数类的训练样本:" + str(shuju_1))
        print("多数类的训练样本:" + str(len(project_y) - shuju_1))
        k_loss = []
        k_auc = []
        tem_cmd=n_estimators  #*****************************************
        canshuming="n_estimators"
        for k in tem_cmd:  # 对参数进行控制，选择参数表现好的，可视化展示
            print("k:"+str(k))
            clf = RandomForestClassifier(n_estimators=k) #******************
            auc = cross_val_score(clf, project_X, project_y, cv=10, scoring='roc_auc')  # for classification   精度
            aloss = -cross_val_score(clf, project_X, project_y, cv=10,
                                    scoring='neg_mean_squared_error')  # for regression    损失函数
            k_auc.append(auc.mean())  # 计算均值得分
            k_loss.append(aloss.mean())
        plt.subplot(3,1,j+1)
        if j==0:
            plt.title(u"超参数"+canshuming+"在数据集CMC不平衡时AUC值",fontproperties=myfont)
        plt.plot(tem_cmd, k_auc)
        for a, b in zip(tem_cmd, k_auc):
            plt.text(a, b + 0.00001, '%.3f' % b, ha='center', va='bottom', fontsize=9)
        plt.ylabel(u"auc值"+"(少数类占比0."+str(zhanbi[j])+")",fontproperties=myfont)
    plt.xlabel("Value of "+canshuming+" for " + sf)

    plt.savefig("../Data/picture/CMC/" +sf+"_"+canshuming+".png")
    plt.show()
if __name__ == '__main__':
    suanfa=["LR","KNN","SVM","GNB","DT","MLP","AdaBT","GBDT","RF"]
    index=8
    zhanbi=[2,5,8]
    # for i in zhanbi:
    #     # filename = "../Data/interData/zhanbi"+str(i)+"/train.csv"
    ceping(zhanbi,"a",suanfa[index])




