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
seg_CTid = seg['CTtime0id_[0, 12)'].tolist()
seg_CTid.extend(seg['CTtime1id_[12, 72)'].tolist())
seg_CTid.extend(seg['CTtime2id_[72, 336)'].tolist())
seg_name = seg['CTtime0_[0, 12)'].tolist()
seg_name.extend(seg['CTtime1_[12, 72)'].tolist())
seg_name.extend(seg['CTtime2_[72, 336)'].tolist())

# get train test name and id
train_id_used, train_name_used = [], []
for id in train_CTid:
    if id in seg_CTid:
        train_id_used.append(id)
        idx_id = seg_CTid.index(id)
        train_name_used.append(seg_name[idx_id])
    if id not in seg_CTid:
        print(f"this train id {id} not in all data")

test_id_used, test_name_used = [], []
for id in test_CTid:
    if id in seg_CTid:
        test_id_used.append(id)
        idx_id = seg_CTid.index(id)
        test_name_used.append(seg_name[idx_id])
    if id not in seg_CTid:
        print(f"this test id {id} not in all data")

# complment length for list 
max_lt = max([len(train_id_used), len(test_id_used)])
for i in range(max_lt - len(train_id_used)):
    train_id_used.append(None)
for i in range(max_lt - len(train_name_used)):
    train_name_used.append(None)
for i in range(max_lt - len(test_name_used)):
    test_name_used.append(None)
for i in range(max_lt - len(test_id_used)):
    test_id_used.append(None)

# # save to excel file
# data_used = pd.DataFrame(train_name_used, columns=["train_name"])
# data_used['train_CTid'] = train_id_used
# data_used["test_name"] = test_name_used
# data_used['test_CTid'] = test_id_used
# data_used.to_excel('./0721/train_test_split.xlsx', index=False)

# get name in needed data but not in train and test
all_lab_name = train_name_used + test_name_used
all_need_name = seg_name + seg['withoutCTid_[0, 12)'].tolist() + seg['withoutCTid_[12, 72)'].tolist() + seg['withoutCTid_[72, 336)'].tolist()
all_not_name = [x for x in all_need_name if x not in all_lab_name and not (isinstance(x, float) and math.isnan(x))]
print(f"train have but needed names: {all_not_name}\nnumber of names: {len(all_not_name)}")