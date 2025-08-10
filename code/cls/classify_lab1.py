import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from datetime import datetime
from utils.pre4data import lasso_dimension_reduction, if_same
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils.util import get_next_result_folder,  save_results, calculate_metrics


def main():
    data_train_path = 'D:/thrid_beijing_hospital_data/0804lab1-train.xlsx'
    data_test_path  = 'D:/thrid_beijing_hospital_data/0804lab1-test.xlsx'
    base_dir = 'D:/thrid_beijing_hospital_data/results_0804_lab1'
    ct_mode = data_train_path.split('-')[0].split('/')[-1]
    data_train = pd.read_excel(data_train_path)
    data_test  = pd.read_excel(data_test_path)
    lab_describe = 'cpc1-2=0_cpc3-5=1_lab1'
    data_train = data_train.drop(columns=['CTid', 'name'])
    train_nse = data_train[['nse极值', 'nse极值差']]
    data_test = data_test.drop(columns=['CTid', 'name'])
    test_nse = data_test[['nse极值', 'nse极值差']]
    nseif = True
    
    # without cpc5  ------->  means dead people data
    # train_df1 = train_df1[train_df1['CPC'] != 5]
    # test_df1 = test_df1[test_df1['CPC'] != 5]

    # ============================================= set dataset cpc split =============================================
    # CPC 1-2 --> label 0,  CPC3-5 --> label 1
    data_train['label'] = pd.cut(data_train['CPC'], bins=[0, 2, 5], labels=[0, 1])
    data_test['label']  = pd.cut(data_test['CPC'],  bins=[0, 2, 5], labels=[0, 1])

    # CPC 1-4 --> label 0,  CPC5   --> label 1
    # train_df1['label'] = pd.cut(train_df1['CPC'], bins=[0, 4, 5], labels=[0, 1])
    # test_df1['label'] = pd.cut(test_df1['CPC'], bins=[0, 4, 5], labels=[0, 1])

    # CPC 1-2 --> label 0,  CPC3-4 --> label 1
    # train_df1['label'] = pd.cut(train_df1['CPC'], bins=[0, 2, 4], labels=[0, 1])
    # test_df1['label'] = pd.cut(test_df1['CPC'], bins=[0, 2, 4], labels=[0, 1])
    # ============================================= set dataset cpc split =============================================

    data_train = data_train.drop(['CPC'], axis=1) 
    data_test = data_test.drop(['CPC'], axis=1) 

    # dataset 
    print(f"训练集形状: {data_train.shape}, 训练集时间: {ct_mode}")
    result_folder = get_next_result_folder(base_path=base_dir)

    y_index_s = data_train.iloc[:, -1]
    scale_pos_weight = len(y_index_s[y_index_s == 0]) / len(y_index_s[y_index_s == 1])
    scaler = StandardScaler()
    print(f"数据划分完毕")

    # 初始化分类器
    ratio0 = len([x for x in data_train['label'].tolist() if x == 0]) / len(data_train['label'].tolist())
    ratio1 = len([x for x in data_train['label'].tolist() if x == 1]) / len(data_train['label'].tolist())
    clf = GaussianNB(priors=[ratio0, ratio1])    # priors=[0.5, 0.5]
    clf = LogisticRegression()

    lgbm_params = {
        'objective': 'binary',       # 二分类任务
        'metric': 'binary_logloss',  # 使用logloss作为评价指标
        'learning_rate': 0.016,
        'max_depth': 6,
        'n_estimators': 500,
        'subsample': 0.7,            # 构建每棵树时使用的样本比例
        'colsample_bytree': 0.7,     # 每棵树使用的特征比例
        'scale_pos_weight': scale_pos_weight,  # 根据数据不平衡调整正负样本的权重
        'random_state': 42,
    }
    clf = LGBMClassifier(**lgbm_params)

    catboost_params = {
        'iterations': 500,           # 迭代次数
        'depth': 6,                  # 树的深度
        'learning_rate': 0.01,       # 学习率
        'loss_function': 'Logloss',  # 损失函数
        'eval_metric': 'AUC',        # AUC作为评价指标
        'scale_pos_weight': scale_pos_weight,  # 样本不平衡的调整
        'random_seed': 42,
        'verbose': 0                 # 不输出训练过程
    }
    clf = CatBoostClassifier(**catboost_params)

    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': ['logloss'],
        'learning_rate': 0.016,
        'max_depth': 6,
        'n_estimators': 600,
        'subsample': 0.72,           # 用于构建每棵树的样本比例
        'colsample_bytree': 0.705,   # 控制每棵树在构建时使用的特征比例
        # 'scale_pos_weight': 1.4,   # 根据实际正负样本比例设置权重 len(y[y==0]) / len(y[y==1]),
        'gamma': 0.1,
        'min_child_weight': 1,       # 降低以增加模型灵活性
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        # 'tree_method': 'hist',  
        # 'device': 'cuda',
    }
    clf = xgb.XGBClassifier(**xgb_params)

    clf = SVC(
        kernel='rbf',              # 使用RBF核函数
        C=10,                      # 正则化参数
        gamma='auto',              # scale
        probability=True,          # 启用概率估计
        class_weight='balanced',   # 处理类别不平衡 
        random_state=42)    
                                                    
    random_var = [42, 46, 52]

    # 参数网格
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 6, 7],
        'n_estimators': [100, 200, 500],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }
    mode = clf.__class__.__name__
    parameter_clf = clf.get_params()
    print(f"classifier is: {mode}")
    acc_scores = []
    recall_scores = []
    specificity_scores = []
    precision_scores = []
    npv_scores = []
    auc_scores = []
    selected_features_all = []
    save_path = os.path.join(base_dir, 'roc_curve_{}_time_{}_{}_{}'.format(mode, ct_mode, lab_describe, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    os.makedirs(save_path, exist_ok=True)

    metrics_history = {'ACC': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'NPV': [], 'AUC': [], }
    random_states = 1307

    results = "lab: {}\nclassifier mode is: {}\n\nrandom_states: {}, train num: {}, test num: {}\n".format(lab_describe, mode, random_states, len(data_train), len(data_test))
    X_train = data_train.drop('label', axis=1)
    y_train = data_train['label']
    X_test = data_test.drop('label', axis=1)
    y_test = data_test['label']

    # save progress
    results += "\ntrain index: \n{}\ntest index: \n{}\n".format(X_train.index.tolist(), X_test.index.tolist())
    results += f"train: label 0 num: {y_train.values.tolist().count(0)} ratio: {y_train.values.tolist().count(0) / len(y_train):.4f}, label 1 num: {y_train.values.tolist().count(1)} ratio: {y_train.values.tolist().count(1) / len(y_train):.4f}\n"
    results += f"test : label 0 num: {y_test.values.tolist().count(0)} ratio: {y_test.values.tolist().count(0) / len(y_test):.4f}, label 1 num: {y_test.values.tolist().count(1)} ratio: {y_test.values.tolist().count(1) / len(y_test):.4f}\n\n\n"

    # 特征降维
    print(f"data train shape is : {X_train.shape}")
    data_train_lasso, selected_features, best_alphas = lasso_dimension_reduction(data_train)
    if nseif:
        print(f"with nse data training...")
        results+=f"with nse data training..."
        data_train_lasso = pd.concat([data_train_lasso, train_nse], axis=1)
        cols = [c for c in data_train_lasso.columns if c != 'label'] + ['label']
        data_train_final = data_train_lasso[cols]

    else:
        print(f"without nse data training...")
        results+=f"without nse data training..."

    X_train = data_train_lasso.iloc[:, :-1] 
    y_train = data_train_lasso.iloc[:, -1]
    print(f"data train shape is : {X_train.shape}")


    print(f"data selected features num is : {len(selected_features)}")
    X_test = X_test[selected_features]

    t = if_same(X_train, X_test)
    if t:
        print(f"X_train and X_val is same")
    else:
        print(f"X_train is not same as X_val")
    print(f"X_test.shape: {X_test.shape}, X_train.shape: {X_train.shape}")

    # 标准化
    X_train_scaled = scaler.fit_transform(X_train)  
    X_val_scaled = scaler.transform(X_test)

    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 0] = len(y_train) / (2 * (y_train == 0).sum())
    sample_weights[y_train == 1] = len(y_train) / (2 * (y_train == 1).sum())

    print(f"训练开始")
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)         #贝叶斯分类器 , logic分类器
    # clf.fit(X_train_scaled, y_train)                                     #SVM, xgb, lgbm, catboost分类器
    print(f"训练完成")

    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled)[:, -1]

    ACC, Recall, Specificity, Precision, NPV, roc_auc = calculate_metrics(y_test, y_pred, y_prob, save_roc_path=save_path, mode=mode)

    acc_scores.append(ACC)
    recall_scores.append(Recall)
    specificity_scores.append(Specificity)
    precision_scores.append(Precision)
    npv_scores.append(NPV)
    auc_scores.append(roc_auc) 
    selected_features_all.append(f"number: {len(selected_features)}, features: {selected_features}\n")
            
    final_ACC = np.mean(acc_scores)
    final_Recall = np.mean(recall_scores)
    final_Specificity = np.mean(specificity_scores)
    final_Precision = np.mean(precision_scores)
    final_NPV = np.mean(npv_scores)
    final_AUC = np.mean(auc_scores)
    print("\n最终测试集的具体指标值:")
    print(f"准确率 (ACC): {final_ACC:.3f} ± {np.std(acc_scores):.3f}")
    print(f"召回率 (Recall): {final_Recall:.3f} ± {np.std(recall_scores):.3f}")
    print(f"特异性 (Specificity): {final_Specificity:.3f} ± {np.std(specificity_scores):.3f}")
    print(f"精确率 (PPV): {final_Precision:.3f} ± {np.std(precision_scores):.3f}")
    print(f"阴性预测值 (NPV): {final_NPV:.3f}  ± {np.std(npv_scores):.3f}")
    print(f"AUC值: {final_AUC:.3f} ± {np.std(auc_scores):.3f}")

    final_results = {
        'Recall': f"{final_Recall:.3f} ± {np.std(recall_scores):.3f}",
        'Specificity': f"{final_Specificity:.3f} ± {np.std(specificity_scores):.3f}",
        'ACC': f"{final_ACC:.3f} ± {np.std(acc_scores):.3f}",
        'PPV': f"{final_Precision:.3f} ± {np.std(precision_scores):.3f}",
        'NPV': f"{final_NPV:.3f}  ± {np.std(npv_scores):.3f}",
        'AUC': f"{final_AUC:.3f} ± {np.std(auc_scores):.3f}\n",
    }
    for metric, value in final_results.items():
        results += f"\n{metric}: {value}\n"
    for feature in selected_features_all:
        results += f"{feature}\n"
    results += "\n\n"
    for pm in parameter_clf:
        results += "\nparameter: {} \n{} values is: {}\n".format(pm, pm, parameter_clf[pm])
    save_results(results, result_folder)

if __name__ == "__main__":
    main()