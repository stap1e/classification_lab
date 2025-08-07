import pandas as pd

i_list = ['CPC', 'CTid']
nse_path = "D:/PycharmProject/classification/datasets/CRF_name.xlsx"
t0_features_path = "D:/PycharmProject/classification/t0_s123_features_all.xlsx"
t1_train_path = './t1_s123_nse_train.xlsx'
t0_train_path = './t0_s123_nse_train.xlsx'
t1_features_path = "D:/PycharmProject/classification/t1_s123_features_all.xlsx"
nse = pd.read_excel(nse_path, sheet_name=0, header=0)
t0_features = pd.read_excel(t0_train_path, sheet_name=0, header=0)
t1_features = pd.read_excel(t1_train_path, sheet_name=0, header=0)

t0_features['label'] = pd.cut(t0_features['CPC'], bins=[0, 4, 5], labels=[1, 0])
t1_features['label'] = pd.cut(t1_features['CPC'], bins=[0, 4, 5], labels=[1, 0])

t0_features_final = t0_features.drop(columns=i_list, axis=1).copy()
t1_features_final = t1_features.drop(columns=i_list, axis=1).copy()
t0_features_final.to_excel('t0_ctfeatures_all_train.xlsx', index=False)
t1_features_final.to_excel('t1_ctfeatures_all_train.xlsx', index=False)

t0_all = t0_features.copy()
t1_all = t1_features.copy()
   
# 修改nse的列名
nse = nse.rename(columns={"姓名": "name"})                             #  df = df.rename(columns={"旧列名": "新列名"})
nse_sorted_t0 = t0_all[['name']].merge(nse, on='name', how='inner')
nse_sorted_t1 = t1_all[['name']].merge(nse, on='name', how='inner')
nse_sorted_t0['label'] = pd.cut(nse_sorted_t0['CPC分'], bins=[0, 4, 5], labels=[1, 0])
nse_sorted_t1['label'] = pd.cut(nse_sorted_t1['CPC分'], bins=[0, 4, 5], labels=[1, 0])
# nse_sorted_t0.to_excel('t0_nse_sorted.xlsx', index=False)
# nse_sorted_t1.to_excel('t1_nse_sorted.xlsx', index=False)

# for col in nse.columns:
#     if col not in t0_all.columns and col not in ['序号', '姓名', '性别', '年龄', '结局（0：死亡 1：存活）','CPC分']:
#         t0_all[col] = nse_sorted_t0[col]
#     if col not in t1_all.columns and col not in ['序号', '姓名', '性别', '年龄', '结局（0：死亡 1：存活）','CPC分']:
#         t1_all[col] = nse_sorted_t1[col]


# 分箱操作
# t0_all['label'] = pd.cut(t0_all['CPC'], bins=[0, 4, 5], labels=[1, 0])
# t1_all['label'] = pd.cut(t1_all['CPC'], bins=[0, 4, 5], labels=[1, 0])
# t0_nse_final = t0_all.drop(columns=i_list, axis=1).copy()
# t1_nse_final = t0_all.drop(columns=i_list, axis=1).copy()
# t0_nse_final.to_excel('t0_nse_all_train.xlsx', index=False)
# t1_nse_final.to_excel('t1_nse_all_train.xlsx', index=False)

print(f"a")









