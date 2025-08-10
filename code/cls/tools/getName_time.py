import pandas as pd
import math

train_path = './0721/0728traindata.xlsx'
test_path = './0721/0728testdata.xlsx'
seg_path = './0721/need_ctid_latest.xlsx'

seg = pd.read_excel(seg_path)
train = pd.read_excel(train_path)
test = pd.read_excel(test_path)

train_CTid = train['CTid'].tolist()
test_CTid = test['CTid'].tolist()

seg_CTid1 = seg['CTtime0id_[0, 12)'].tolist()
seg_CTid2 = seg['CTtime1id_[12, 72)'].tolist()
seg_CTid3 = seg['CTtime2id_[72, 336)'].tolist()
seg_name1 = seg['CTtime0_[0, 12)'].tolist()
seg_name2 = seg['CTtime1_[12, 72)'].tolist()
seg_name3 = seg['CTtime2_[72, 336)'].tolist()

train_id_used1, train_name_used1 = [], []
train_id_used2, train_name_used2 = [], []
train_id_used3, train_name_used3 = [], []
test_id_used1, test_name_used1 = [], []
test_id_used2, test_name_used2 = [], []
test_id_used3, test_name_used3 = [], []

# get train test name and id for each segment
for id in train_CTid:
    if id in seg_CTid1:
        train_id_used1.append(id)
        idx_id = seg_CTid1.index(id)
        train_name_used1.append(seg_name1[idx_id])
    elif id in seg_CTid2:
        train_id_used2.append(id)
        idx_id = seg_CTid2.index(id)
        train_name_used2.append(seg_name2[idx_id])
    elif id in seg_CTid3:
        train_id_used3.append(id)
        idx_id = seg_CTid3.index(id)
        train_name_used3.append(seg_name3[idx_id])
    else:
        print(f"this train id {id} not in all data")

# get test name and id for each segment
for id in test_CTid:
    if id in seg_CTid1:
        test_id_used1.append(id)
        idx_id = seg_CTid1.index(id)
        test_name_used1.append(seg_name1[idx_id])
    elif id in seg_CTid2:
        test_id_used2.append(id)
        idx_id = seg_CTid2.index(id)
        test_name_used2.append(seg_name2[idx_id])
    elif id in seg_CTid3:
        test_id_used3.append(id)
        idx_id = seg_CTid3.index(id)
        test_name_used3.append(seg_name3[idx_id])
    else:
        print(f"this test id {id} not in all data")

# complement length for lists
max_lt = max(len(train_id_used1), len(test_id_used1), len(train_id_used2), len(test_id_used2), len(train_id_used3), len(test_id_used3))
for i in range(max_lt - len(train_id_used1)):
    train_id_used1.append(None)
for i in range(max_lt - len(train_name_used1)):
    train_name_used1.append(None)
for i in range(max_lt - len(test_name_used1)):
    test_name_used1.append(None)
for i in range(max_lt - len(test_id_used1)):
    test_id_used1.append(None)

for i in range(max_lt - len(train_id_used2)):
    train_id_used2.append(None)
for i in range(max_lt - len(train_name_used2)):
    train_name_used2.append(None)
for i in range(max_lt - len(test_name_used2)):
    test_name_used2.append(None)
for i in range(max_lt - len(test_id_used2)):
    test_id_used2.append(None)

for i in range(max_lt - len(train_id_used3)):
    train_id_used3.append(None)
for i in range(max_lt - len(train_name_used3)):
    train_name_used3.append(None)
for i in range(max_lt - len(test_name_used3)):
    test_name_used3.append(None)
for i in range(max_lt - len(test_id_used3)):
    test_id_used3.append(None)

# save to excel file
data_used = pd.DataFrame({
    "train_name[0, 12)": train_name_used1,
    "train_name[12, 72)": train_name_used2,
    "train_name[72, 336)": train_name_used3,
    "train_id[0, 12)": train_id_used1,
    "train_id[12, 72)": train_id_used2,
    "train_id[72, 336)": train_id_used3,
    "test_name[0, 12)": test_name_used1,
    "test_name[12, 72)": test_name_used2,
    "test_name[72, 336)": test_name_used3,
    "test_id[0, 12)": test_id_used1,
    "test_id[12, 72)": test_id_used2,
    "test_id[72, 336)": test_id_used3,
})
data_used.to_excel('./0721/ByTime_train_test_splits.xlsx', index=False)