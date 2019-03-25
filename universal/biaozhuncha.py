# -*- coding: utf-8 -*-
"""
@file: biaozhuncha.py
@author: tianfeihan
@time: 2019-03-24  11:35:05
@description: 
"""
import xlrd
from xlutils.copy import copy
import numpy
import xlwt
import os
import universal.GlobalVariable as gv
def Cal(project_name, excel_filename,savefile):
    data=xlrd.open_workbook(excel_filename)
    std_data=[]
    bainyi_data=[]
    for i in range(9):
        table=data.sheets()[i]
        auc=table.col_values(4)[1:]
        std=numpy.std(auc,ddof=1)
        std_data.append(std)
        ave=numpy.mean(auc)
        # print(ave)
        bainyi=std/ave
        bainyi_data.append(bainyi)
    print(std_data)
    print("======")
    print(bainyi_data)
    if os.path.exists(savefile):
        excel_data = xlrd.open_workbook(savefile)
        table = excel_data.sheet_by_name("bzc")
        sheet1 = excel_data.sheet_by_name("byz")
        rows = table.nrows
        newWB = copy(excel_data)
        sheet = newWB.get_sheet(0)
        sheet2 = newWB.get_sheet(1)
        sheet.write(rows, 0, project_name)
        sheet2.write(rows, 0, project_name)
        for j in range(1,10):
            sheet.write(rows, j, std_data[j-1])
            sheet2.write(rows, j, bainyi_data[j - 1])
        newWB.save(savefile)

    else:
        workbook = xlwt.Workbook()
        table = workbook.add_sheet("bzc")
        sheet = workbook.add_sheet("byz")
        table.write(0, 0, "数据集")
        table.write(1, 0, project_name)
        sheet.write(0, 0, "数据集")
        sheet.write(1, 0, project_name)
        for index in range(1,gv.alg_num+1):
            table.write(0,index , gv.alg_name[index-1])
            table.write(1,index,std_data[index-1])
            sheet.write(0, index, gv.alg_name[index - 1])
            sheet.write(1, index, bainyi_data[index - 1])
        workbook.save(savefile)



if __name__ == '__main__':
    # for i in range(len(gv.project)):
    #     project_name=gv.project[i]
    #     excel_filename="../Data/excel_GYH/"+project_name+".xls"
    #     DrwBox(project_name,excel_filename)
    #
    for  i in range(12):
        project_name = gv.project[i]
        excel_filename = "../Data/excel/" + project_name + ".xls"
        sfile="../Data/interData/biaozhuncha.xls"
        Cal(project_name, excel_filename,sfile)
    # project_name = "MR"
    # excel_filename = "../Data/excel/" + project_name + ".xls"
    # sfile = "../Data/interData/biaozhuncha.xls"
    # Cal(project_name, excel_filename, sfile)