# -*- coding: utf-8 -*-
"""
@file: jifen.py
@author: tianfeihan
@time: 2019-03-24  20:12:02
@description: 
"""
import xlrd
from xlutils.copy import copy
import numpy
import xlwt
import os
import universal.GlobalVariable as gv
def xihua(b):
    res=[0]*len(b)
    for i in range(len(b)):
        index=b.index(min(b))
        b[index]=100
        res[index]=(9-i)
    return res




def jifenFun(a):
    jieguo=[]

    for i in range(len(a[0])):
        tem_a=[]
        for j in range(len(a)):
            tem_a.append(a[j][i])
        temp=xihua(tem_a)
        jieguo.append(temp)
    return jieguo
def jifenFunCal(a):
    jieguo=[]
    for i in range(len(a)):
        temp=xihua(a[i])
        jieguo.append(temp)
    return jieguo


def calca(project_name, excel_filename, savefile):
    data = xlrd.open_workbook(excel_filename)
    auc_data=[]
    for i in range(9):
        table=data.sheets()[i]
        auc=table.col_values(4)[1:]
        auc_data.append(auc)
    # auc_data=numpy.ndarray(auc_data)
    # auc_data.reshape(auc_data,(12,9))
    jiegu=jifenFun(auc_data)
    print(jiegu)
    if os.path.exists(savefile):
        excel_data = xlrd.open_workbook(savefile)
        table = excel_data.sheet_by_name("jifen")
        rows = table.nrows
        newWB = copy(excel_data)
        sheet = newWB.get_sheet(0)
        for i  in range(len(jiegu)):
            for  j in range(len(jiegu[0])):
                sheet.write(rows+i+2, j, jiegu[i][j])
        # for j in range(1,10):
        #     sheet.write(rows, j, std_data[j-1])
        newWB.save(savefile)

    else:
        workbook = xlwt.Workbook()
        table = workbook.add_sheet("jifen")

        for index in range(gv.alg_num):
            table.write(0,index , gv.alg_name[index])
        for i  in range(len(jiegu)):
            for  j in range(len(jiegu[0])):
                table.write(i+1, j, jiegu[i][j])

        workbook.save(savefile)
def calcau( excel_filename, sfile):
    data = xlrd.open_workbook(excel_filename)
    bzc = []
    qyz=[]
    table = data.sheets()[0]
    for i in range(1,13):
        auc = table.row_values(i)[1:10]
        bzc.append(auc)
    jiegu = jifenFunCal(bzc)
    sheet = data.sheets()[1]
    for i in range(1, 13):
        auc = sheet.row_values(i)[1:10]
        qyz.append(auc)
    jiegu1 = jifenFunCal(qyz)
    if os.path.exists(sfile):
        os.remove(sfile)
    else:
        pass
    workbook = xlwt.Workbook()
    table = workbook.add_sheet("jifen")

    for index in range(gv.alg_num):
        table.write(0, index, gv.alg_name[index])
    for i in range(len(jiegu)):
        for j in range(len(jiegu[0])):
            table.write(i + 1, j, jiegu[i][j]+jiegu1[i][j])

    workbook.save(sfile)



if __name__ == '__main__':
    # for i in range(12):
    #     project_name =gv.project[i]
    #     excel_filename = "../Data/excel/" + project_name + ".xls"
    #     sfile = "../Data/interData/jifen.xls"
    #     calca(project_name, excel_filename, sfile)

    excel_filename = "../Data/interData/biaozhuncha.xls"
    sfile = "../Data/interData/byz_jifen.xls"
    calcau( excel_filename, sfile)
