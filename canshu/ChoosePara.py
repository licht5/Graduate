# -*- coding: utf-8 -*-
"""
@file: ChoosePara.py
@author: tianfeihan
@time: 2019-03-27  11:18:25
@description: 从待选的超参数选择随不平衡程度变化变化的参数
"""

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
import universal.globalVariable as gv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
# filename="../dataFile/mushroom.csv"

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def ceping(filename,att):
    train_filename = gv.train_filename
    test_filename = gv.test_filename
    project_X, project_y = shujuchuli.getdata(train_filename, gv.att_add, gv.att_type)
    ts_data, ts_target = shujuchuli.getdata(test_filename, gv.att_add, gv.att_type)
    if not gv.flag:
        shuju_1 = sum(project_y)
        print("种类1的训练样本:" + str(shuju_1))
        print("种类2的训练样本:" + str(len(project_y) - shuju_1))

        shuju_2 = sum(ts_target)
        print("种类1的测试样本:" + str(shuju_2))
        print("种类2的测试样本:" + str(len(ts_target) - shuju_2))
        gv.flag=True


    # return train_data,ts_data,train_target,ts_target

    # project_data=np.loadtxt(filename,delimiter=",")
    # if (att=="a"):
    #     project_X=project_data[:,:-1]
    #     project_y=project_data[:,-1]
    # else:
    #     project_X = project_data[:, 1:]
    #     project_y = project_data[:, 0]
    # print("总数目："+str(len(project_y))+"\n为1数量："+str(len(project_y)-project_y.sum()))


    # ============= LR ================
    # tol=np.arange(5e-5,2e-4,1e-5)  #[5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,11e-5,12e-5,13e-5,14e-5,16e-5]
    # C=range(5,20,1)
    intercept_scaling=np.arange(0.1,3,0.2)
    random_state=range(1,30,2)
    # solver=["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
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
    # gamma=[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,5,10,10]
    gamma=np.arange(0.001,0.031,0.002)
    max_iter=range(-5,5,1)
    decision_function_shape=["ovo", "ovr"]

    # ============= nb ================
    var_smoothing=np.arange(1e-9,1e-8,1e-9)

    # ============= tree ================
    max_depth=range(1,10,1)
    min_samples_split=range(160,200,3)
    # min_samples_split = np.arange(0.001, 0.1, 0.003)
    min_samples_leaf=range(25,40,1)
    min_weight_fraction_leaf=np.arange(0,0.5,0.1)
    max_leaf_nodes=range(2,10,1)


    # algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute']
    learning_rate=np.arange(0.01,0.2,0.01)
    # max_depth=range(1,10,1)
    # min_samples_split=range(2,30,1)
    # min_samples_leaf=[0.01,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]

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
    learning_rate=np.arange(0.5,2,0.1)
    algorithm=["SAMME","SAMME.R"]


    # ============= gbdt ================
    loss=["deviance", "exponential"]
    learning_rate=np.arange(0.01,0.08,0.01)
    n_estimators=range(10,30,2)
    subsample=np.arange(0.1,1,0.1)
    # min_samples_split=range(2,10,1)
    # min_samples_leaf=range(1,10,1)
    max_depth=range(1,10,1)

    # =============== rdforest==========
    n_estimators=range(5,30,1)
    max_depth=range(1,10,1)
    min_samples_split=range(35,50,1)
    max_leaf_nodes=range(2,30,5)

    k_loss = []
    k_auc = []
    tem_cmd=max_leaf_nodes
    for k in tem_cmd:  # 对参数进行控制，选择参数表现好的，可视化展示
        print(k)
        knn = RandomForestClassifier(max_leaf_nodes=k)
        auc = cross_val_score(knn, project_X, project_y, cv=10, scoring='roc_auc')  # for classification   精度
        loss = -cross_val_score(knn, project_X, project_y, cv=10,
                                    scoring='neg_mean_squared_error')  # for regression    损失函数
        k_auc.append(auc.mean())  # 计算均值得分
        k_loss.append(loss.mean())

    plt.plot(tem_cmd, k_auc)
    plt.xlabel("Value of max_leaf_nodes for "+ bl.suanfa)
    plt.ylabel("Cross-validates roc_auc")
    plt.title(bl.project_name)
    for a, b in zip(tem_cmd, k_auc):
        plt.text(a, b + 0.0001, '%.4f' % b, ha='center', va='bottom', fontsize=9)
    # plt.savefig('../picture/saunfa/'+i+"_"+bl.project_name+'_estimator.png')
    add = "/Users/tianfeihan/Desktop/breast_mass/" + bl.suanfa + "/"
    plt.savefig(add+"max_leaf_nodes.png")
    plt.show()

    #
    # clf=tree.DecisionTreeClassifier(min=6)
    # print("canshu:"+str(clf.min_samples_split))
    # the_auc = cross_val_score(clf, project_X, project_y, cv=10, scoring='roc_auc')  # for classification   精度
    # print(the_auc.mean())




if __name__ == '__main__':
    bl.suanfa="RdT"
    bl.filename = "../canshuData/" + bl.project_name + ".csv"
    ceping(bl.filename,"a")




