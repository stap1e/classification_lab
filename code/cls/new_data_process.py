import pandas as pd
import os

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

# dropdata and dropdata0 是不需要的特征, 且这些特征在进行降维时的值会导致无法读取, 字符串类型或者其他不可使用的类型
dropdata = ['diagnostics_Image-original_Hash', 'diagnostics_Imag e-original_Hash_1', 'diagnostics_Image-original_Hash_2', 'diagnostics_Image-original_Hash_3',
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
            'CPC_1', 'CPC_2', 'CPC_3', 'CTid_1', 'CTid_2', 'CTid_3', 'name_1', 'name_2', 'name_3',
            'diagnostics_Mask-corrected_Spacing', 'diagnostics_Mask-corrected_Size', 'diagnostics_Mask-corrected_BoundingBox', 'diagnostics_Mask-corrected_VoxelNum',
            'diagnostics_Mask-corrected_VolumeNum','diagnostics_Mask-corrected_CenterOfMassIndex',
            'diagnostics_Mask-corrected_CenterOfMass', 'diagnostics_Mask-corrected_Mean', 'diagnostics_Mask-corrected_Minimum',
            'diagnostics_Mask-corrected_Maximum', 'live', 'diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
            'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Dimensionality']

dropdata0 = ['diagnostics_Mask-corrected_Spacing', 'diagnostics_Mask-corrected_Size', 'diagnostics_Mask-corrected_BoundingBox', 
             'diagnostics_Mask-corrected_VoxelNum', 'diagnostics_Mask-corrected_VolumeNum', 
             'diagnostics_Mask-corrected_CenterOfMassIndex', 'diagnostics_Mask-corrected_CenterOfMass', 
             'diagnostics_Mask-corrected_Mean', 'diagnostics_Mask-corrected_Minimum', 'diagnostics_Mask-corrected_Maximum', 'live']

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