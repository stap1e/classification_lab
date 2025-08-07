import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from utils.pre4data import plot_metrics, lasso_dimension_reduction, get_importantfeature_xgb, if_same
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from classify import get_importantfeature_xgb, get_next_result_folder, save_results, calculate_metrics

def main():
    t0_train_path = "D:/PycharmProject/classification/t0_ctfeatures_all_train.xlsx"
    t1_train_path = "D:/PycharmProject/classification/t1_ctfeatures_all_train.xlsx"
    t0_nse_path = "D:/PycharmProject/classification/t0_nse_sorted.xlsx"
    t1_nse_path = "D:/PycharmProject/classification/t1_nse_sorted.xlsx"
    result_folder = get_next_result_folder()
    
    t0_train = pd.read_excel(t0_train_path)
    t0_nse = pd.read_excel(t0_nse_path)
    t1_train = pd.read_excel(t1_train_path)
    t1_nse = pd.read_excel(t1_nse_path)
    print(f"训练集形状:t0: {t0_train.shape}, t1: {t1_train.shape}")
    
    X0_index = t0_train.iloc[:, :-1]
    y0_index = t0_train.iloc[:, -1]
    label0_index = t0_train.iloc[:, -1]
    X1_index = t1_train.iloc[:, :-1]
    y1_index = t1_train.iloc[:, -1]
    label1_index = t1_train.iloc[:, -1]
    
    
    scale_pos_weight0 = len(label0_index[label0_index == 0]) / len(label0_index[label0_index == 1])
    scaler0 = StandardScaler()
    scale_pos_weight1 = len(label1_index[label1_index == 0]) / len(label1_index[label1_index == 1])
    scaler1 = StandardScaler()
    print(f"数据划分完毕")
    
    best_model = None
    best_scaler = None
    best_auc = 0
    last_featurs = []
    
    # 初始化分类器
    # clf = GaussianNB()  #priors=[0.5, 0.5]
    # clf = LogisticRegression()
    
    lgbm_params = {
    'objective': 'binary',  # 二分类任务
    'metric': 'binary_logloss',  # 使用logloss作为评价指标
    'learning_rate': 0.016,
    'max_depth': 6,
    'n_estimators': 500,
    'subsample': 0.7,  # 构建每棵树时使用的样本比例
    'colsample_bytree': 0.7,  # 每棵树使用的特征比例
    'scale_pos_weight': scale_pos_weight1,  # 根据数据不平衡调整正负样本的权重
    'random_state': 42,
    }
    # clf = LGBMClassifier(**lgbm_params)
    
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': ['logloss'],
        'learning_rate': 0.04,
        'max_depth': 6,
        'n_estimators': 600,
        'subsample': 0.72,         # 用于构建每棵树的样本比例
        'colsample_bytree': 0.705,  # 控制每棵树在构建时使用的特征比例
        # 'scale_pos_weight': 1.4,   # 根据实际正负样本比例设置权重 len(y[y==0]) / len(y[y==1]),
        'gamma': 0.1,
        'min_child_weight': 1,     # 降低以增加模型灵活性
        'scale_pos_weight': scale_pos_weight1,
        'random_state': 4,
        # 'tree_method': 'hist',  
        # 'device': 'cuda',
    }
    # clf = xgb.XGBClassifier(**xgb_params)
    
    clf = SVC(
        kernel='rbf',              # 使用RBF核函数
        C=1,                     # 正则化参数
        gamma='auto',              # scale
        probability=True,          # 启用概率估计
        class_weight='balanced', # 处理类别不平衡 
        random_state=42
    )                                                    
    random_var = [42, 46, 52]
    
    mode = clf.__class__.__name__
    print(mode)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    acc_scores = []
    recall_scores = []
    specificity_scores = []
    precision_scores = []
    npv_scores = []
    auc_scores = []
    selected_features_all = []
    save_path = './results_nse/roc_curve'
    os.makedirs(save_path, exist_ok=True)
    metrics_history = {
        'ACC': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'NPV': [], 'AUC': [], 
    }

    
    for fold, (train_index, test_index) in enumerate(kf.split(X1_index, y1_index), 1):
        print(f"==============第 {fold} 折交叉验证==============")
        # 特征降维
        data_train =  t1_train
        print(f"data shape is : {data_train.shape}")
        data_nse = t1_nse
        data_train_1 = data_train.iloc[train_index]
        nse_train = data_nse.iloc[train_index]
        nse_test = data_nse.iloc[test_index]
        print(f"data for {fold} 折 train shape is : {data_train_1.shape}")
        data_train_lasso, selected_features, best_alphas = lasso_dimension_reduction(data_train_1)
        # selected_features = selected_features[:45]
        # data_train_lasso, selected_features = get_importantfeature_xgb(data_train)
        X_train = data_train_lasso.iloc[:, :-1]
        y_train = data_train_lasso.iloc[:, -1]

        data_val = data_train.iloc[test_index]
        X_val = data_val.iloc[:, :-1]
        y_val = data_val.iloc[:, -1]
        X_val = X_val[selected_features]
        
        for col in nse_train.columns:
            if col not in X_train.columns and col not in ['序号', '姓名', '性别', '年龄', '结局（0：死亡 1：存活）','CPC分', 'label']:
                X_train[col] = nse_train[col]
        for col in nse_test.columns:
            if col not in X_val.columns and col not in ['序号', '姓名', '性别', '年龄', '结局（0：死亡 1：存活）','CPC分', 'label']:
                X_val[col] = nse_test[col]
                
        t = if_same(X_train, X_val)
        if t:
            print(f"X_train and X_val is same")
        else:
            print(f"X_train is not same as X_val")
        # y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        print(f"第 {fold} 折交叉验证X_test.shape: {X_val.shape}, X_train.shape: {X_train.shape}")

        # 标准化
        X_train_scaled = scaler0.fit_transform(X_train)  
        X_val_scaled = scaler0.transform(X_val)

        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = len(y_train) / (2 * (y_train == 0).sum())
        sample_weights[y_train == 1] = len(y_train) / (2 * (y_train == 1).sum())

        print(f"训练开始")
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)         #贝叶斯分类器 , logic分类器
        # clf.fit(X_train_scaled, y_train)   #SVM, xgb, lgbm, catboost分类器
        print(f"训练完成")

        y_pred = clf.predict(X_val_scaled)
        y_prob = clf.predict_proba(X_val_scaled)[:, -1]

        ACC, Recall, Specificity, Precision, NPV, roc_auc = calculate_metrics(y_val, y_pred, y_prob, save_roc_path=save_path, mode=mode, fold=fold)

        acc_scores.append(ACC)
        recall_scores.append(Recall)
        specificity_scores.append(Specificity)
        precision_scores.append(Precision)
        npv_scores.append(NPV)
        auc_scores.append(roc_auc)
        # selected_features_all.append(selected_features)
        selected_features_all.append({"折数": fold, "features": X_train.columns.to_list()})
        selected_features_all.append("\n")

        print(f"第 {fold} 折交叉验证ACC:{ACC:.3f}")
        print(f"第 {fold} 折交叉验证Recall:{Recall:.3f}")
        print(f"第 {fold} 折交叉验证Specificity:{Specificity:.3f}")
        print(f"第 {fold} 折交叉验证Precision:{Precision:.3f}")
        print(f"第 {fold} 折交叉验证NPV:{NPV:.3f}")
        print(f"第 {fold} 折交叉验证AUC:{roc_auc:.3f}")

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = clf
            best_scaler = scaler0
            
    # X_test = X_test[last_features]
    # X_test_scaled = best_scaler.transform(X_test)
    # y_pred = best_model.predict(X_test_scaled)
    # y_prob = best_model.predict_proba(X_test_scaled)[:, -1]
    # final_ACC, final_Recall, final_Specificity, final_Precision, final_NPV, final_AUC = calculate_metrics(y_test, y_pred, y_prob)
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
        'ACC': f"{final_ACC:.3f} ± {np.std(acc_scores):.3f}",
        'Recall': f"{final_Recall:.3f} ± {np.std(recall_scores):.3f}",
        'Specificity': f"{final_Specificity:.3f} ± {np.std(specificity_scores):.3f}",
        'PPV': f"{final_Precision:.3f} ± {np.std(precision_scores):.3f}",
        'NPV': f"{final_NPV:.3f}  ± {np.std(npv_scores):.3f}",
        'AUC': f"{final_AUC:.3f} ± {np.std(auc_scores):.3f}",
        'resluts':f"{selected_features_all}\nmode is: {mode}"
    }
    results = "\n"
    for metric, value in final_results.items():
        results += f"{metric}: {value}\n"
    save_results(results, result_folder)
    # plot_metrics(metrics_history, result_folder)
        
    # except Exception as e:
    #     print(f"发生错误: {str(e)}")
if __name__ == "__main__":
    main()


