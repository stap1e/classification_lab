import pandas as pd
import os
import numpy as np

from utils.pre4data import lasso_dimension_reduction, analyze_columns, get_importantfeature_xgb, drop_columns

os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
nse_path = 'D:/PycharmProject/classification/nse_0418.xlsx'
ct_time0_path = "D:/PycharmProject/classification/datasets/time0_use(addonerow).xlsx"
ct_time1_path = "D:/PycharmProject/classification/datasets/time1_use(addonerow).xlsx"
ct_time0_sheet0_path = "D:/PycharmProject/classification/datasets/time0_new_c00.xlsx"  # D:\PycharmProject\classification\datasets\time0_new_c00.xlsx
name_ctid_path = "D:/PycharmProject/classification/datasets/CTSeg.xlsx"

dropdata = ['diagnostics_Image-original_Hash', 'diagnostics_Image-original_Hash_1', 'diagnostics_Image-original_Hash_2', 'diagnostics_Image-original_Hash_3',
                'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Hash_1', 'diagnostics_Mask-original_Hash_2', 'diagnostics_Mask-original_Hash_3',
                'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Spacing_1', 'diagnostics_Image-original_Spacing_2','diagnostics_Image-original_Spacing_3',
                'diagnostics_Image-original_Size', 'diagnostics_Image-original_Size_1', 'diagnostics_Image-original_Size_2', 'diagnostics_Image-original_Size_3',
                'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Spacing_1', 'diagnostics_Mask-original_Spacing_2', 'diagnostics_Mask-original_Spacing_3',
                'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_Size_1', 'diagnostics_Mask-original_Size_2', 'diagnostics_Mask-original_Size_3',
                'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_BoundingBox_1', 'diagnostics_Mask-original_BoundingBox_2', 'diagnostics_Mask-original_BoundingBox_3',
                'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMassIndex_1', 'diagnostics_Mask-original_CenterOfMassIndex_2', 'diagnostics_Mask-original_CenterOfMassIndex_3',
                'diagnostics_Mask-original_CenterOfMass', 'diagnostics_Mask-original_CenterOfMass_1', 'diagnostics_Mask-original_CenterOfMass_2', 'diagnostics_Mask-original_CenterOfMass_3',
                'diagnostics_Mask-original_BoundingBox.1', 'diagnostics_Mask-original_BoundingBox.1_1', 'diagnostics_Mask-original_BoundingBox.1_2', 'diagnostics_Mask-original_BoundingBox.1_3',  
                'diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_BoundingBox_1', 'diagnostics_Mask-original_BoundingBox_2', 'diagnostics_Mask-original_BoundingBox_3',
                'CPC_1', 'CPC_2', 'CPC_3', 'CTid_1', 'CTid_2', 'CTid_3', 'name_1', 'name_2', 'name_3']

# 合并time0,1数据，读取Nse数据
cttime0 = pd.read_excel(ct_time0_path, sheet_name=['sheet1', 'sheet2', 'sheet3'])  # use sheet 0,1,2,3
cttime1 = pd.read_excel(ct_time1_path, sheet_name=['sheet1', 'sheet2', 'sheet3'])  # use sheet 1,2,3
ct_time0_sheet0 = pd.read_excel(ct_time0_sheet0_path)
# cttime0['sheet0'] = ct_time0_sheet0


# 提取共同姓名
name_ctid = pd.read_excel(name_ctid_path)
nse = pd.read_excel(nse_path, sheet_name=0, header=0)
nse_name = nse['姓名']
time0_ids, time1_ids = [], []
for name in nse_name:
    name_list = name_ctid[name_ctid['name'] == '{}'.format(name)]
    time0_name = name_list[name_list['CTtime']==0]
    if len(time0_name) > 0:
        time0_ids.append(time0_name.iloc[0, 0])
    time1_name = name_list[name_list['CTtime']==1]
    if len(time1_name) > 0:
        time1_ids.append(time1_name.iloc[0, 0])

    
# cttime0['sheet0'] = cttime0['sheet0'][cttime0['sheet0']['CTid'].isin(time0_ids)]
cttime0['sheet1'] = cttime0['sheet1'][cttime0['sheet1']['CTid'].isin(time0_ids)]
cttime0['sheet2'] = cttime0['sheet2'][cttime0['sheet2']['CTid'].isin(time0_ids)]   
cttime0['sheet3'] = cttime0['sheet3'][cttime0['sheet3']['CTid'].isin(time0_ids)]  

cttime1['sheet1'] = cttime1['sheet1'][cttime1['sheet1']['CTid'].isin(time1_ids)]
cttime1['sheet2'] = cttime1['sheet2'][cttime1['sheet2']['CTid'].isin(time1_ids)]   
cttime1['sheet3'] = cttime1['sheet3'][cttime1['sheet3']['CTid'].isin(time1_ids)]

# 合并time0 sheet1,2,3特征
ct0_1 = cttime0['sheet1'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_1': cttime0['sheet1'][x].copy() for x in cttime0['sheet1'].columns}
cttime0['sheet1'] = pd.concat([cttime0['sheet1'], pd.DataFrame(new_columns)], axis=1)
cttime0['sheet1'] = cttime0['sheet1'].drop(columns=ct0_1, axis=1)

ct0_2 = cttime0['sheet2'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_2': cttime0['sheet2'][x].copy() for x in cttime0['sheet2'].columns}
cttime0['sheet2'] = pd.concat([cttime0['sheet2'], pd.DataFrame(new_columns)], axis=1)
cttime0['sheet2'] = cttime0['sheet2'].drop(columns=ct0_2, axis=1)

ct0_3 = cttime0['sheet3'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_3': cttime0['sheet3'][x].copy() for x in cttime0['sheet3'].columns}
cttime0['sheet3'] = pd.concat([cttime0['sheet3'], pd.DataFrame(new_columns)], axis=1)
cttime0['sheet3'] = cttime0['sheet3'].drop(columns=ct0_3, axis=1)

cttime0['sheet1'] = drop_columns(cttime0['sheet1'], dropdata) 
cttime0['sheet2'] = drop_columns(cttime0['sheet2'], dropdata)        
cttime0['sheet3'] = drop_columns(cttime0['sheet3'], dropdata)
print(f"{'='*15}正在合并time0 sheet1,2,3特征{'='*15}")
t0_feature = cttime0['sheet1']
count0, count1 = 0, 0
for j in range(0, cttime0['sheet1'].shape[1]):
    if cttime0['sheet2'].columns[j] not in t0_feature.columns:
        x = cttime0['sheet2'].columns[j]
        count0 += 1
        t0_feature = pd.concat([t0_feature, cttime0['sheet2'][x]], axis=1)
    if cttime0['sheet3'].columns[j] not in t0_feature.columns:
        y = cttime0['sheet3'].columns[j]
        count1 += 1
        t0_feature = pd.concat([t0_feature, cttime0['sheet3'][y]], axis=1)
print(f"{'='*15}合并time0 sheet1,2,3特征完成, count0:{count0}, count1:{count1}{'='*15}")
c_pro0 = t0_feature.columns.drop(['CPC', 'CTid']).to_list()
print(f"\n{'='*30} 列数据分析 {'='*30}")
t0_features = t0_feature[c_pro0]
same_cols = []
for col in t0_features.columns:
    unique_values = t0_features[col].unique()
    if len(unique_values) == 1:
        same_cols.append(col)
print(f"总列数: {len(t0_features.columns)}")
t0_features = t0_feature.drop(columns=same_cols, axis=1)
print(f"去除列内值完全相同的列数: {len(same_cols)}")
t0_features.to_excel("t0_s123_nse_new_processed.xlsx", index=False)


# 合并time1 sheet1,2,3特征
ct1_1 = cttime1['sheet1'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_1': cttime1['sheet1'][x].copy() for x in cttime1['sheet1'].columns}
cttime1['sheet1'] = pd.concat([cttime1['sheet1'], pd.DataFrame(new_columns)], axis=1)
cttime1['sheet1'] = cttime1['sheet1'].drop(columns=ct1_1, axis=1)

ct1_2 = cttime1['sheet2'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_2': cttime1['sheet2'][x].copy() for x in cttime1['sheet2'].columns}
cttime1['sheet2'] = pd.concat([cttime1['sheet2'], pd.DataFrame(new_columns)], axis=1)
cttime1['sheet2'] = cttime1['sheet2'].drop(columns=ct1_2, axis=1)

ct1_3 = cttime1['sheet3'].columns.drop(['CPC', 'CTid'])
new_columns = {x + '_3': cttime1['sheet3'][x].copy() for x in cttime1['sheet3'].columns}
cttime1['sheet3'] = pd.concat([cttime1['sheet3'], pd.DataFrame(new_columns)], axis=1)
cttime1['sheet3'] = cttime1['sheet3'].drop(columns=ct1_3, axis=1)

cttime1['sheet1'] = drop_columns(cttime1['sheet1'], dropdata) 
cttime1['sheet2'] = drop_columns(cttime1['sheet2'], dropdata)        
cttime1['sheet3'] = drop_columns(cttime1['sheet3'], dropdata)
print(f"{'='*15}正在合并time1 sheet1,2,3特征{'='*15}")
t1_feature = cttime1['sheet1']
count0, count1 = 0, 0
for j in range(0, cttime1['sheet1'].shape[1]):
    if cttime1['sheet2'].columns[j] not in t1_feature.columns:
        x = cttime1['sheet2'].columns[j]
        count0 += 1
        t1_feature = pd.concat([t1_feature, cttime1['sheet2'][x]], axis=1)
    if cttime1['sheet3'].columns[j] not in t1_feature.columns:
        y = cttime1['sheet3'].columns[j]
        count1 += 1
        t1_feature = pd.concat([t1_feature, cttime1['sheet3'][y]], axis=1)
print(f"{'='*15}合并time1 sheet1,2,3特征完成, count0:{count0}, count1:{count1}{'='*15}")
c_pro1 = t1_feature.columns.drop(['CPC', 'CTid']).to_list()
print(f"\n{'='*30} 列数据分析 {'='*30}")
t1_features = t1_feature[c_pro1]
same_cols = []
for col in t1_features.columns:
    unique_values = t1_features[col].unique()
    if len(unique_values) == 1:
        same_cols.append(col)
print(f"总列数: {len(t1_features.columns)}")
t1_features = t1_feature.drop(columns=same_cols, axis=1)
print(f"去除列内值完全相同的列数: {len(same_cols)}")
t1_features.to_excel("t1_s123_nse_new_processed.xlsx", index=False)
# print("ceshi")




