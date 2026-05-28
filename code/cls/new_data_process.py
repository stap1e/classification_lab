import pandas as pd
import os

from config.feature_columns import RADIOMICS_DROP_COLUMNS
from utils.pre4data import drop_columns
from sklearn.model_selection import train_test_split

# set data path
data0_path = './0721/result_0.xlsx'
data1_path = './0721/result_1_n.xlsx'
data2_path = './0721/result_2.xlsx'
# data3_path = './new_data/radiomics_result_output.xlsx'
data_path = './0721/0728data_delete_n.xlsx'
data_withid_path = './0721/0728data_withCTid.xlsx'

data0_1 = pd.read_excel(data0_path, sheet_name=1)
data1_1 = pd.read_excel(data1_path, sheet_name=1)
data2_1 = pd.read_excel(data2_path, sheet_name=1)
# data3_1 = pd.read_excel(data3_path, sheet_name=1)

data0_2 = pd.read_excel(data0_path, sheet_name=2)
data1_2 = pd.read_excel(data1_path, sheet_name=2)
data2_2 = pd.read_excel(data2_path, sheet_name=2)
# data3_2 = pd.read_excel(data3_path, sheet_name=2)

data0_3 = pd.read_excel(data0_path, sheet_name=3)
data1_3 = pd.read_excel(data1_path, sheet_name=3)
data2_3 = pd.read_excel(data2_path, sheet_name=3)
# data3_3 = pd.read_excel(data3_path, sheet_name=3)

# data_1 = pd.concat([data0_1, data1_1, data2_1, data3_1])
data_1 = pd.concat([data0_1, data1_1, data2_1])

# 不需要的 diagnostics / 元数据列（见 config/feature_columns.py）
dropdata = RADIOMICS_DROP_COLUMNS

data0_1 = drop_columns(data0_1, dropdata)
data1_1 = drop_columns(data1_1, dropdata)
data2_1 = drop_columns(data2_1, dropdata)
# data3_1 = drop_columns(data3_1, dropdata)

data0_2 = drop_columns(data0_2, dropdata)
data1_2 = drop_columns(data1_2, dropdata)
data2_2 = drop_columns(data2_2, dropdata)
# data3_2 = drop_columns(data3_2, dropdata)

data0_3 = drop_columns(data0_3, dropdata)
data1_3 = drop_columns(data1_3, dropdata)
data2_3 = drop_columns(data2_3, dropdata)
# data3_3 = drop_columns(data3_3, dropdata)

# merge diffierent patient data
data1 = pd.concat([data0_1, data1_1, data2_1], ignore_index=True)
data2 = pd.concat([data0_2, data1_2, data2_2], ignore_index=True)
data3 = pd.concat([data0_3, data1_3, data2_3], ignore_index=True)

# merge diffierent sheet data for data1, data2, data3
feature1 = data1.columns.drop(['CPC'])
feature1_withid = data1.columns.drop(['CPC', 'CTid'])
new_columns = {x + '_1': data1[x].copy() for x in data1.columns}
data1 = pd.concat([data1, pd.DataFrame(new_columns)], axis=1)
data1_withid = data1.drop(columns=feature1_withid, axis=1)
data1_withid = data1_withid.drop(columns=['CPC_1', 'CTid_1'], axis=1)
data1 = data1.drop(columns=feature1, axis=1)
data1 = data1.drop(columns=['CPC_1', 'CTid_1'], axis=1)

feature2 = data2.columns.drop(['CPC', 'CTid'])
new_columns = {x + '_2': data2[x].copy() for x in data2.columns}
data2 = pd.concat([data2, pd.DataFrame(new_columns)], axis=1)
data2 = data2.drop(columns=feature2, axis=1)
data2 = data2.drop(columns=['CPC_2', 'CTid_2', 'CPC', 'CTid'], axis=1)

feature3 = data3.columns.drop(['CPC', 'CTid'])
new_columns = {x + '_3': data3[x].copy() for x in data3.columns}
data3 = pd.concat([data3, pd.DataFrame(new_columns)], axis=1)
data3 = data3.drop(columns=feature3, axis=1)
data3 = data3.drop(columns=['CPC_3', 'CTid_3', 'CPC', 'CTid'], axis=1)

data = pd.concat([data1, data2, data3], axis=1)
data_withCTid = pd.concat([data1_withid, data2, data3], axis=1)
train_df, test_df = train_test_split(
        data_withCTid,
        test_size=0.2,                  # 20% 作为测试集
        stratify=data_withCTid['CPC'],  # 分层抽样
        random_state=42)
# data_withCTid.to_excel(data_withid_path, index=False)
# data.to_excel(data_path, index=False)
train_df.to_excel('./0721/0728traindata.xlsx', index=False)
test_df.to_excel('./0721/0728testdata.xlsx', index=False)
print(f"successfully save latest train data to {data_path}")
num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0
for i in data['CPC'].tolist():
    if i == 1:
        num1+=1
    if i ==2:
        num2+=1
    if i==3:
        num3 +=1
    if i==4:
        num4 +=1
    if i==5:
        num5+=1
print(f"1: {num1}\n2: {num2}\n3: {num3}\n4: {num4}\n5: {num5}")