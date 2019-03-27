# -*- coding: utf-8 -*-
"""
@file: ParaMain.py
@author: tianfeihan
@time: 2019-03-25  20:33:46
@description: 参数优化的入口函数
"""
import os
import numpy
import xlrd
import xlwt
from xlutils.copy import copy

from universal import GlobalVariable as gv
from coreCode import ParaOptimize
from coreCode import SetDataUnbalanced


def func1(filename,att,minority,excel_filename,minority_num,cishu):
    if minority==0.05 and os.path.exists(excel_filename):
        os.remove(excel_filename)
    elif minority<0.05:
        print("********************** wrong：不平衡率设置不对 ********************")
    else:
        pass
    for i in range(cishu):
        SetDataUnbalanced.SetDataUnbalancedFunc(filename, gv.test_rate,  att,minority,minority_num)
        ParaOptimize.totalAlgrithon(att)
        if (i == (cishu-1)):
            gv.flag = False



def write_excel(excel_filename,data):
    print("========== run to write_excel====== ")
    data_tem = []
    for i in range(gv.alg_num):
        tem = numpy.array(data[i])
        tem_mean = tem.mean(axis=0).tolist()
        data_tem.append(tem_mean)
    if os.path.exists(excel_filename):
        # excel_data = xlrd.open_workbook(gv.excel_filename)
        for inj in range(gv.alg_num):
            excel_data = xlrd.open_workbook(excel_filename)
            table = excel_data.sheet_by_name(gv.alg_name[inj])
            rows = table.nrows
            newWB = copy(excel_data)
            sheet = newWB.get_sheet(inj)
            for j in range(gv.evaluation_num):
                sheet.write(rows, j, data_tem[inj][j])
            newWB.save(excel_filename)
    else:
        workbook = xlwt.Workbook()
        for index in range(gv.alg_num):
            table = workbook.add_sheet(gv.alg_name[index])
            for e_num in range(gv.evaluation_num):
                table.write(0, e_num, gv.evaluation[e_num])
            for data_index in range(gv.evaluation_num):
                table.write(1,data_index,data_tem[index][data_index])
        workbook.save(excel_filename)

if __name__ == '__main__':
    att = "a"
    index=2
    filename = "../Data/rawData/" + gv.project[index] + ".csv"
    excel_filename = "../Data/Paraexcel/" + gv.project[index] + "_para.xls"
    minority_num = gv.MINORITY_num[index]
    print(filename)
    print(excel_filename)
    print(minority_num)
    for j in range(len(gv.MINORITY_RATIO)):
        minority = gv.MINORITY_RATIO[j]
        print(minority)
        func1(filename, att, minority, excel_filename, minority_num, 2)
        write_excel(excel_filename, gv.algrithm)

