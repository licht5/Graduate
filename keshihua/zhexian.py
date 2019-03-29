# -*- coding: utf-8 -*-
"""
@file: zhexian.py
@author: tianfeihan
@time: 2019-03-22  19:25:19
@description: 绘制折线图
"""
import numpy as np
from pylab import mpl
import pandas as pd
import matplotlib
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import xlrd
import universal.GlobalVariable as gv
def DrwZhexian(project_name,excel_filename):
    myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    data = xlrd.open_workbook(excel_filename)
    color = ["#8B4726", "#ff9900", "#ff0000", "#ff00cc", "#99ff00",
             "#6600cc","#00cc99","#0033cc","#000033"]
    style_line=[">","<","|","+",".","x","o","s","*"]
    label=("LR","KNN","SVM","GNB","DT","MLP","AdaBT","GBDT","RF")

    x_label=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
             0.75,0.8,0.85,0.9]
    carV=locals()
    for i in range(gv.alg_num):

        table = data.sheet_by_name(gv.alg_name[i])
        # ncols = table.nrows
        # x = np.arange(1, ncols)
        data_ = table.col_values(4, start_rowx=1)
        carV["l"+str(i)]=plt.plot(x_label,data_,c=color[i],marker=style_line[i],label=label[i])




    plt.xlabel(u'少数类比例', fontproperties=myfont)
    plt.ylabel(u'AUC', fontproperties=myfont)
    plt.title(u'' + project_name + '数据集各算法AUC值折线图' + '', fontproperties=myfont)

    plt.legend((carV["l0"][0],carV["l1"][0], carV["l2"][0], carV["l3"][0],carV["l4"][0],carV["l5"][0],
                carV["l6"][0], carV["l7"][0],carV["l8"][0]), label)
    plt.savefig('../Data/picture/zhexian/'+project_name+'_para.png')
    plt.show()
if __name__ == '__main__':
    # for i in range(len(gv.project)-1):
    #     project_name=gv.project[i]
    #     excel_filename="../Data/Paraexcel/"+project_name+".xls"
    #     DrwZhexian(project_name,excel_filename)

    project_name = "CMC"
    excel_filename = "../Data/Paraexcel/" + project_name + "_para.xls"
    DrwZhexian(project_name, excel_filename)

