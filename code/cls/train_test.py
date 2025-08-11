# split train and test
import pandas as pd
from sklearn.model_selection import train_test_split


lab1_data_path   = 'D:/thrid_beijing_hospital_data/0804lab1-CTdata_withCTidname_nse.xlsx'
lab1_train_path  = 'D:/thrid_beijing_hospital_data/0804lab1-train.xlsx'
lab1_test_path   = 'D:/thrid_beijing_hospital_data/0804lab1-test.xlsx'


data_lab1 = pd.read_excel(lab1_data_path)
train_lab1, test_lab1 = train_test_split(
        data_lab1,
        test_size=0.2,               # 20% 作为测试集  4:1(8:2)
        stratify=data_lab1['CPC'],   # 分层抽样
        random_state=42)
# train_lab1.to_excel(lab1_train_path, index=False)
# test_lab1.to_excel(lab1_test_path, index=False)

print(f"a")