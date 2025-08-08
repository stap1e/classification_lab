"""_summary_
lab1: use 159(160) data with nse feature in fixed train-test-set to train

lab2: use 101(114) data in 5-fold to train with cross-validation and nse feature

Statistic:
data with nse: 170
lab1 data    : 160    lab1 data with nse: 159
lab2 data    : 114    lab2 data with nse: 101
"""

import numpy as np
import pandas as pd
import math, os, openpyxl

def ColoredifFill(name_col, data, ws, expname='time'):
    coloredif_list = []
    for row in ws.iter_rows(min_row=2, min_col=name_col, max_col=name_col):   # openpyxl列号从1开始
        cell = row[0]    # 可以取多个列
        # 判断是否有填充色
        has_color = cell.fill.fgColor.type == 'rgb' and cell.fill.fgColor.rgb == 'FFFFFF00'  # '00000000'  无底色,  'FFFFFF00' 填充为黄色
        coloredif_list.append(has_color)
    data[f'{expname}_{name_col}'] = coloredif_list
    return data

def getNameId(data, namelist, idlist, savename, saveid, expname):
    for l in namelist:
        idx = namelist.index(l)
        iflist = data[f'{expname}_{l+1}'].tolist()
        namedata = data.iloc[:, l].tolist()
        iddata   = data.iloc[:, idlist[idx]].tolist()
        savename.extend([name for name, flag in zip(namedata, iflist) if not flag])
        saveid.extend([id for id, flag in zip(iddata, iflist) if not flag])
    
    savename = [x for x in savename if not pd.isna(x)]
    saveid   = [x for x in saveid   if not pd.isna(x)]
    return savename, saveid

nse_path             = "D:/thrid_beijing_hospital_data/0804数据-极值、极值差.xlsx"
data_path            = "D:/thrid_beijing_hospital_data/result_0805_deleted.xlsx"
match_path           = "D:/thrid_beijing_hospital_data/ByTime_train_test_splits_0804.xlsx"
save_mid_path        = "D:/thrid_beijing_hospital_data/split_mid.xlsx"
need_path            = 'D:/thrid_beijing_hospital_data/0804数据-新增12小时内统计结果.xlsx'
lab2_error_path      = 'D:/thrid_beijing_hospital_data/error.xlsx'
lab2_nameid_path     = 'D:/thrid_beijing_hospital_data/0805-ct.xlsx'
save_train_test_path = 'D:/thrid_beijing_hospital_data/0808data/train_test.xlsx'
save_fordoctor_path  = 'D:/thrid_beijing_hospital_data/0808data/train_test_withname_Bytime.xlsx'

nse_data   = pd.read_excel(nse_path)
ct_data    = pd.read_excel(data_path)
match_data = pd.read_excel(match_path)
error_data = pd.read_excel(lab2_error_path)
lab2_all   = pd.read_excel(lab2_nameid_path)
lab2_all_id = lab2_all['CTid'].tolist()
lab2_all_name = lab2_all['name'].tolist()
error_id   = error_data['error_id'].tolist()
wb = openpyxl.load_workbook(match_path)
ws = wb.active
need0 = openpyxl.load_workbook(need_path, read_only=True)['NSE统计表']
need_name0 = pd.read_excel(need_path, sheet_name=1)

ct_use_id    = ct_data['CTid'].tolist()
nse_use_data = nse_data[nse_data['nse极值'].notna()]
nse_use_name = nse_use_data['姓名（标绿的排除）'].tolist()

expname = 'time'
match_data1 = ColoredifFill(1, match_data,  ws, expname=expname)    # first variable means columns selected and you colored
match_data2 = ColoredifFill(2, match_data1, ws, expname=expname)
match_data3 = ColoredifFill(3, match_data2, ws, expname=expname)
match_data7 = ColoredifFill(7, match_data3, ws, expname=expname)

# get name, id in train_test_split file
match_name, match_id, need_namecol, need_idcol = [], [], [0, 1, 2, 6], [3, 4, 5, 9]
match_name, match_id = getNameId(match_data7, need_namecol, need_idcol, match_name, match_id, expname)
match_name.extend(match_data.iloc[:, 7].tolist())
match_name.extend(match_data.iloc[:, 8].tolist())
match_name = [x for x in match_name if not pd.isna(x)]
match_id.extend(match_data.iloc[:, 10].tolist())
match_id.extend(match_data.iloc[:, 11].tolist())
match_id = [x for x in match_id if not pd.isna(x)]

# get blue name for extract ct feature for zpy
blueif = []
idx = 0
for row in need0.iter_rows(min_row=2, min_col=2, max_col=2):
    cell = row[0]
    idx += 1
    has_color = cell.fill.fgColor.type == 'theme' and cell.fill.fgColor.theme == 8 if cell.value is not None  else False         # 8 blue, 9 green
    blueif.append(has_color)
    # if idx >= 120:

need_name = need_name0['姓名（标绿的排除）'].tolist()
need_name = need_name[:-1]
get_bluename = [x for x, flag in zip(need_name, blueif) if flag]        


# get lab1 data
# in 160 but not in nse use name=纪淑贞, id= s668982-0002-00002-000001
lab1_name = [x for x in match_name if x in nse_use_name]
lab1_id   = [id for name, id in zip(match_name, match_id) if name in lab1_name]


# get lab2 data
# in 114 but not in nse use name=['孙文姬', '汪九如', '王炳安', '范玉珍', '赵庆书', '马宁', '陈青岚', '董秀珍', '胡龙凤', '李来青', '陈志明', '景宝峰', '王艳艳']
lab2_name = [x for x in get_bluename if x in nse_use_name]
# lab2_if   = [id for name, id in zip()]

print(f"a")