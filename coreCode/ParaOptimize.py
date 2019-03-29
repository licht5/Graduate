# -*- coding: utf-8 -*-
"""
@file: ParaOptimize.py
@author: tianfeihan
@time: 2019-03-25  20:21:57
@description: 最优实验的算法部分
"""

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
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import warnings

warnings.filterwarnings('ignore')
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


def showResultAUC(result, i, ts_target):
    auc = metrics.roc_auc_score(ts_target, result)
    print("auc:" + str(auc))
# ******************************* 逻辑斯特回归 *******************************
def LogistRe(train_data,ts_data, train_target,ts_target):
    clf = LogisticRegression()
    param_grid = {'C': np.arange(0.01,100), 'intercept_scaling': np.arange(0.1,10),
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============LogistRe=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = LogisticRegression(C=best_parameters['C'], intercept_scaling=best_parameters['intercept_scaling'],
                             solver=best_parameters['solver'], random_state=0)
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = LogisticRegression()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc=metrics.roc_auc_score(ts_target,result)
    auc1=metrics.roc_auc_score(ts_target, result1)
    if auc1>auc:
        showResult(result1, 0, ts_target)
    else:
        showResult(result, 0, ts_target)




def KNNClass(train_data,ts_data, train_target,ts_target):
    clf = KNeighborsClassifier()
    param_grid = {'leaf_size': range(10, 41, 5), 'n_neighbors': range(3, 16, 3), 'algorithm': ['auto', "brute"],
                  'p': np.arange(1, 5.1, 0.5)}
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============KNNClass=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = KNeighborsClassifier(leaf_size=best_parameters['leaf_size'], n_neighbors=best_parameters['n_neighbors'],
                               algorithm=best_parameters['algorithm'], p=best_parameters['p'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = KNeighborsClassifier()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 1, ts_target)
    else:
        showResult(result, 1, ts_target)





# *******************************支持向量机 70/345*******************************
def svcFunc(train_data,ts_data, train_target,ts_target):
    clf = svm.SVC(probability=True)
    param_grid = {'C': np.arange(0.1, 2, 0.2), 'gamma': np.arange(0.001,1,0.001), 'max_iter': [-2, -1, 0, 1]}
                  # 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============svcFunc=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = svm.SVC(C=best_parameters['C'], gamma=best_parameters['gamma'],
                  max_iter=best_parameters['max_iter'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = svm.SVC()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 2, ts_target)
    else:
        showResult(result, 2, ts_target)


#*******************************高斯朴素贝叶斯 77/345*******************************
def gsNB(train_data,ts_data, train_target,ts_target):

    gnb = GaussianNB()
    param_grid ={'var_smoothing':np.arange(1e-10,2e-9)}
    # grid_search = GridSearchCV(gnb, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(gnb, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============gsNB=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = GaussianNB(var_smoothing=best_parameters['var_smoothing'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)
    showResult(result, 3, ts_target)



#******************************* 决策树*******************************
def deTree(train_data,ts_data, train_target,ts_target):
    clf = tree.DecisionTreeClassifier()
    # 需要搜索的超参数及待选区间
    max_depth_para=list(range(1, 10))
    max_depth_para.append(None)
    max_leaf_nodes_para=list(range(2, 30))
    max_leaf_nodes_para.append(None)
    param_grid = {'max_depth': max_depth_para,
                  'max_leaf_nodes': max_leaf_nodes_para,
                  'min_samples_leaf': range(1, 10),
                  'min_samples_split': range(2, 10),
                  'min_weight_fraction_leaf': np.arange(0, 0.5, 0.1)}
    # 随机搜索方法，评价标准为neg_log_loss损失，采用五折交叉验证
    grid_search = RandomizedSearchCV(clf, param_grid, scoring=
    'neg_log_loss', cv=5, n_iter=10, random_state=5)
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search.fit(train_data, train_target)
    # 获得最优的参数集字典数数据并打印显示
    best_parameters = grid_search.best_estimator_.get_params()
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    # 为模型设定超参数
    clf = tree.DecisionTreeClassifier(
        max_depth=best_parameters['max_depth'],
        max_leaf_nodes=best_parameters['max_leaf_nodes'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        min_samples_split=best_parameters['min_samples_split'],
        min_weight_fraction_leaf=best_parameters['min_weight_fraction_leaf'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = tree.DecisionTreeClassifier()
    clf1.fit(train_data,train_target)
    result1 = clf1.predict(ts_data)
    auc = metrics.roc_auc_score(ts_target, result)
    auc1=metrics.roc_auc_score(ts_target, result1)
    if auc>auc1:
        showResult(result, 4, ts_target)
    else:
        showResult(result1, 4, ts_target)




#******************************* 神经网络  *******************************
def MLPClass(train_data,ts_data, train_target,ts_target):
    clf = MLPClassifier()
    # batch_size_para=list(range(100,200))
    # batch_size_para.append('auto')

    param_grid = {'activation': ['logistic','identity’','tanh','relu'], 'alpha': np.arange(5e-5, 14e-5, 1e-5), 'batch_size': range(100, 200),
                  'hidden_layer_sizes': [(i,) for i in range(60, 130)],
                  'learning_rate_init': np.arange(5e-4, 14e-4, 1e-4),
                  'learning_rate': ['constant','invscaling','adaptive'], 'solver': ['lbfgs','sgd','adam']}
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============MLPClass=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = MLPClassifier(activation=best_parameters['activation'],
                        alpha=best_parameters['alpha'],
                        batch_size=best_parameters['batch_size'],
                        hidden_layer_sizes=best_parameters['hidden_layer_sizes'],
                        learning_rate_init=best_parameters['learning_rate_init'],
                        learning_rate=best_parameters['learning_rate'],
                        solver=best_parameters['solver'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = MLPClassifier()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 5, ts_target)
    else:
        showResult(result, 5, ts_target)
    # showResult(result, 5, ts_target)
#******************************* adaboost 的集成方法 *******************************
def adaBoost(train_data,ts_data, train_target,ts_target):
    clf = AdaBoostClassifier()
    param_grid = {'algorithm': ['SAMME','SAMME.R'], 'learning_rate': np.arange(0.5, 1.4, 0.1), 'n_estimators': range(30, 60, 1)}
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============adaBoost=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = AdaBoostClassifier(algorithm=best_parameters['algorithm'],
                             learning_rate=best_parameters['learning_rate'],
                             n_estimators=best_parameters['n_estimators'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = AdaBoostClassifier()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 6, ts_target)
    else:
        showResult(result, 6, ts_target)
    # showResult(result, 6, ts_target)

#******************************* GBDT 的集成方法 *******************************
def GBDT(train_data,ts_data, train_target,ts_target):
    clf = GradientBoostingClassifier()
    param_grid = {'loss': ['deviance','exponential'], 'learning_rate': np.arange(0.01,1), 'max_depth': range(1, 10),
                  'min_samples_split': range(2, 20),
                  'n_estimators': np.arange(60, 120),
                  'subsample': np.arange(0.1, 1.01,0.1)}
    # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============GBDT=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = GradientBoostingClassifier(loss=best_parameters['loss'],
                                     learning_rate=best_parameters['learning_rate'],
                                     max_depth=best_parameters['max_depth'],
                                     min_samples_split=best_parameters['min_samples_split'],
                                     n_estimators=best_parameters['n_estimators'],
                                     subsample=best_parameters['subsample'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = GradientBoostingClassifier()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 7, ts_target)
    else:
        showResult(result, 7, ts_target)
    # showResult(result, 7, ts_target)

# *******************************随机森林的集成方法  n_estimators为5时，准确率最高 *******************************
def rdForest(train_data,ts_data, train_target,ts_target):
    clf = RandomForestClassifier()
    max_depth_para=list(range(2,10))
    max_depth_para.append(None)
    max_leaf_nodes_para=list(range(2,20))
    max_leaf_nodes_para.append(None)
    param_grid = {'max_depth': max_depth_para, 'max_leaf_nodes': max_leaf_nodes_para,
                  'min_samples_split': range(2,20),
                  'n_estimators': range(2,20)}
    # grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
    grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
    grid_search.fit(train_data, train_target)
    best_parameters = grid_search.best_estimator_.get_params()
    print("===============RdForest=======================")
    for patra, val in list(best_parameters.items()):
        print(patra, val)
    clf = RandomForestClassifier(max_depth=best_parameters['max_depth'],
                                 max_leaf_nodes=best_parameters['max_leaf_nodes'],
                                 min_samples_split=best_parameters['min_samples_split'],
                                 n_estimators=best_parameters['n_estimators'])
    clf = clf.fit(train_data, train_target)
    result = clf.predict(ts_data)

    clf1 = RandomForestClassifier()
    clf1 = clf1.fit(train_data, train_target)
    result1 = clf1.predict(ts_data)

    auc = metrics.roc_auc_score(ts_target, result)
    auc1 = metrics.roc_auc_score(ts_target, result1)
    if auc1 > auc:
        showResult(result1, 8, ts_target)
    else:
        showResult(result, 8, ts_target)
    # showResult(result, 8, ts_target)

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





#
# # ******************************* 逻辑斯特回归 *******************************
# def LogistReGrid(train_data,ts_data, train_target,ts_target):
#     clf = LogisticRegression()
#     param_grid = {'C':[0.001,0.01,0.1,1,10,100], 'intercept_scaling': [1,0.1,2,3,10],
#                   'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print('随机搜索-最佳度量值:', grid_search.best_score_)
#     print("===============LogistRe=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = LogisticRegression(C=best_parameters['C'], intercept_scaling=best_parameters['intercept_scaling'],
#                              solver=best_parameters['solver'], random_state=0)
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = LogisticRegression()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc>auc1:
#         showResult(result, 0, ts_target)
#     else:
#         showResult(result1, 0, ts_target)
#
#
# def KNNClassGrid(train_data,ts_data, train_target,ts_target):
#     clf = KNeighborsClassifier()
#     param_grid = {'leaf_size': range(10, 41, 5), 'n_neighbors': range(3, 16, 3), 'algorithm': ['auto', "brute"],
#                   'p': np.arange(1, 3.1, 0.5)}
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============KNNClass=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = KNeighborsClassifier(leaf_size=best_parameters['leaf_size'], n_neighbors=best_parameters['n_neighbors'],
#                                algorithm=best_parameters['algorithm'], p=best_parameters['p'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = KNeighborsClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 1, ts_target)
#     else:
#         showResult(result1, 1, ts_target)
#     # showResult(result, 1, ts_target)
#
#
#
# # *******************************支持向量机 70/345*******************************
# def svcFuncGrid(train_data,ts_data, train_target,ts_target):
#     clf = svm.SVC(probability=True)
#     param_grid = {'C': np.arange(0.4, 1.6, 0.3), 'gamma': ["auto_deprecated"], 'max_iter': [-2, -1, 0, 1]}
#                   # 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5,n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============svcFunc=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = svm.SVC(C=best_parameters['C'], gamma=best_parameters['gamma'],
#                   max_iter=best_parameters['max_iter'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = svm.SVC()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 2, ts_target)
#     else:
#         showResult(result1, 2, ts_target)
#
#
# #*******************************高斯朴素贝叶斯 77/345*******************************
# def gsNBGrid(train_data,ts_data, train_target,ts_target):
#
#     gnb = GaussianNB()
#     param_grid ={'var_smoothing':np.arange(1e-10,2e-9,2e-9)}
#     grid_search = GridSearchCV(gnb, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(gnb, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============gsNB=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = GaussianNB(var_smoothing=best_parameters['var_smoothing'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#     showResult(result, 3, ts_target)
#
#
#
# #******************************* 决策树*******************************
# def deTreeGrid(train_data,ts_data, train_target,ts_target):
#     clf = tree.DecisionTreeClassifier()
#     # 需要搜索的超参数及待选区间
#     max_depth_para=list(range(1, 10,2))
#     max_depth_para.append(None)
#     max_leaf_nodes_para=list(range(2, 30,6))
#     max_leaf_nodes_para.append(None)
#     param_grid = {'max_depth': max_depth_para,
#                   'max_leaf_nodes': max_leaf_nodes_para,
#                   'min_samples_leaf': range(1, 10,2),
#                   'min_samples_split': range(2, 10,2),
#                   'min_weight_fraction_leaf': np.arange(0, 0.5, 0.1)}
#     # 网格搜索方法，评价标准为neg_log_loss损失，采用五折交叉验证
#     # grid_search = GridSearchCV(clf, param_grid, scoring=
#     # 'neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     grid_search.fit(train_data, train_target)
#     # 获得最优的参数集字典数数据
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============deTree=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     # 为模型设定超参数
#     clf = tree.DecisionTreeClassifier(
#         max_depth=best_parameters['max_depth'],
#         max_leaf_nodes=best_parameters['max_leaf_nodes'],
#         min_samples_leaf=best_parameters['min_samples_leaf'],
#         min_samples_split=best_parameters['min_samples_split'],
#         min_weight_fraction_leaf=best_parameters['min_weight_fraction_leaf'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#     # showResult(result, 4, ts_target)
#
#     clf1 = tree.DecisionTreeClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 4, ts_target)
#     else:
#         showResult(result1, 4, ts_target)
#
# #******************************* 神经网络  *******************************
# def MLPClassGrid(train_data,ts_data, train_target,ts_target):
#     clf = MLPClassifier()
#     # batch_size_para=list(range(100,200))
#     # batch_size_para.append('auto')
#
#     # param_grid = {'activation': ['logistic','identity’','tanh','relu'], 'alpha': np.arange(5e-5, 14e-5, 2e-5), 'batch_size': range(100, 200,20),
#     #               'hidden_layer_sizes': [(i,) for i in range(70, 121,10)],
#     #               'learning_rate_init': np.arange(6e-4, 14e-4, 2e-4),
#     #               'learning_rate': ['constant','invscaling','adaptive'], 'solver': ['lbfgs','sgd','adam']}
#     param_grid = {'activation': ['logistic'], 'alpha': np.arange(11e-5, 14e-5, 1e-5), 'batch_size': range(120, 180, 20),
#                   'hidden_layer_sizes': [(i,) for i in range(90, 121, 10)],
#                   'learning_rate_init': np.arange(10e-4, 14e-4, 1e-4),
#                   'learning_rate': ['constant'], 'solver': ['lbfgs']}
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============MLPClass=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = MLPClassifier(activation=best_parameters['activation'],
#                         alpha=best_parameters['alpha'],
#                         batch_size=best_parameters['batch_size'],
#                         hidden_layer_sizes=best_parameters['hidden_layer_sizes'],
#                         learning_rate_init=best_parameters['learning_rate_init'],
#                         learning_rate=best_parameters['learning_rate'],
#                         solver=best_parameters['solver'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = MLPClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 5, ts_target)
#     else:
#         showResult(result1, 5, ts_target)
#     # showResult(result, 5, ts_target)
# #******************************* adaboost 的集成方法 *******************************
# def adaBoostGrid(train_data,ts_data, train_target,ts_target):
#     clf = AdaBoostClassifier()
#     param_grid = {'algorithm': ['SAMME','SAMME.R'], 'learning_rate': np.arange(0.5, 1.2, 0.1), 'n_estimators': range(30, 61, 3)}
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============adaBoost=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = AdaBoostClassifier(algorithm=best_parameters['algorithm'],
#                              learning_rate=best_parameters['learning_rate'],
#                              n_estimators=best_parameters['n_estimators'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = AdaBoostClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 6, ts_target)
#     else:
#         showResult(result1, 6, ts_target)
#     # showResult(result, 6, ts_target)
#
# #******************************* GBDT 的集成方法 *******************************
# def GBDTGrid(train_data,ts_data, train_target,ts_target):
#     clf = GradientBoostingClassifier()
#     param_grid = {'loss': ['deviance','exponential'], 'learning_rate': [0.001,0.01,0.1,1], 'max_depth': range(1, 10,2),
#                   'min_samples_split': range(2, 20,5),
#                   'n_estimators': np.arange(60, 121,20),
#                   'subsample': np.arange(0.6, 1.01,0.1)}
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============GBDT=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = GradientBoostingClassifier(loss=best_parameters['loss'],
#                                      learning_rate=best_parameters['learning_rate'],
#                                      max_depth=best_parameters['max_depth'],
#                                      min_samples_split=best_parameters['min_samples_split'],
#                                      n_estimators=best_parameters['n_estimators'],
#                                      subsample=best_parameters['subsample'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = GradientBoostingClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 7, ts_target)
#     else:
#         showResult(result1, 7, ts_target)
#
#     # showResult(result, 7, ts_target)
#
# # *******************************随机森林的集成方法  n_estimators为5时，准确率最高 *******************************
# def rdForestGrid(train_data,ts_data, train_target,ts_target):
#     clf = RandomForestClassifier()
#     max_depth_para=list(range(2,11,2))
#     max_depth_para.append(None)
#     max_leaf_nodes_para=list(range(2,20,4))
#     max_leaf_nodes_para.append(None)
#     param_grid = {'max_depth': max_depth_para, 'max_leaf_nodes': max_leaf_nodes_para,
#                   'min_samples_split': range(2,20,4),
#                   'n_estimators': range(2,20,4)}
#     grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5)
#     # grid_search = RandomizedSearchCV(clf, param_grid, scoring='neg_log_loss', cv=5, n_iter=10, random_state=5)
#     grid_search.fit(train_data, train_target)
#     best_parameters = grid_search.best_estimator_.get_params()
#     print("===============RdForest=======================")
#     for patra, val in list(best_parameters.items()):
#         print(patra, val)
#     clf = RandomForestClassifier(max_depth=best_parameters['max_depth'],
#                                  max_leaf_nodes=best_parameters['max_leaf_nodes'],
#                                  min_samples_split=best_parameters['min_samples_split'],
#                                  n_estimators=best_parameters['n_estimators'])
#     clf = clf.fit(train_data, train_target)
#     result = clf.predict(ts_data)
#
#     clf1 = RandomForestClassifier()
#     clf1 = clf1.fit(train_data, train_target)
#     result1 = clf1.predict(ts_data)
#     auc = metrics.roc_auc_score(ts_target, result)
#     auc1 = metrics.roc_auc_score(ts_target, result1)
#
#     if auc > auc1:
#         showResult(result, 8, ts_target)
#     else:
#         showResult(result1, 8, ts_target)
#     # showResult(result, 8, ts_target)
# def  totalAlgrithonGrid(att):
#     print("==================="+str(gv.count)+"次算法 =====================")
#     gv.count=gv.count+1
#     train_data,ts_data, train_target,ts_target=getData(att)
#     train_data = preprocessing.scale(train_data)
#     ts_data=preprocessing.scale(ts_data)
#     LogistReGrid(train_data, ts_data, train_target, ts_target)
#     KNNClassGrid(train_data, ts_data, train_target, ts_target)
#     svcFuncGrid(train_data,ts_data, train_target,ts_target)
#     gsNBGrid(train_data,ts_data, train_target,ts_target)
#     deTreeGrid(train_data,ts_data, train_target,ts_target)
#     MLPClassGrid(train_data, ts_data, train_target, ts_target)
#     adaBoostGrid(train_data,ts_data, train_target,ts_target)
#     GBDTGrid(train_data, ts_data, train_target, ts_target)
#     rdForestGrid(train_data,ts_data, train_target,ts_target)