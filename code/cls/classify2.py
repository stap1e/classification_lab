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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

def get_next_result_folder(base_path='D:/PycharmProject/classification/results_nse'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return os.path.join(base_path, 'results_1')
    
    # 查找现有的results_i文件夹
    existing_folders = [d for d in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, d)) 
                       and d.startswith('results_')]
    
    if not existing_folders:
        return os.path.join(base_path, 'results_1')
    
    # 获取现有文件夹的最大编号
    max_num = max([int(f.split('_')[1]) for f in existing_folders])
    
    # 返回下一个编号的文件夹路径
    return os.path.join(base_path, f'results_{max_num + 1}')

def save_results(results_text, result_folder):
    """保存结果到results.txt文件"""
    os.makedirs(result_folder, exist_ok=True)
    
    result_file = os.path.join(result_folder, 'results.txt')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_with_timestamp = f"实验时间: {timestamp}\n\n{results_text}"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(results_with_timestamp)
    
    print(f"\n结果已保存到: {result_file}")

def calculate_metrics(y_true, y_pred, y_prob, save_roc_path=None, mode=None, fold=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    ACC = (tp + tn) / (tp + tn + fp + fn)
    Recall = tp / (tp + fn) if (tp + fn) != 0 else 0  
    Specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  
    Precision = tp / (tp + fp) if (tp + fp) != 0 else 0  
    NPV = tn / (tn + fn) if (tn + fn) != 0 else 0  
    # ROC曲线绘制逻辑
    roc_auc = None
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        
        # 内置绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        if fold:
            plt.savefig(os.path.join(save_roc_path, 'model-{}_fold-{}'.format(mode, fold)), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_roc_path, 'model-{}'.format(mode)), dpi=300, bbox_inches='tight')
        plt.close()

    
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    return ACC, Recall, Specificity, Precision, NPV, roc_auc


def balanced_train_test_split(X, y, test_size=0.4, random_state=None):
    print("初始各类别比例  :", np.unique(y, return_counts=True)[1]/len(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, val_index in sss.split(X, y):
        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]
        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]
    
    # 验证各类别比例是否相同
    print("验证集各类别比例:", np.unique(y_val, return_counts=True)[1]/len(y_val))
    print("训练集各类别比例:", np.unique(y_train, return_counts=True)[1]/len(y_train))
    
    return X_train, X_val, y_train, y_val


def main():
    data_train_path = './0721/0728data_delete.xlsx' 
    ct_mode = data_train_path.split('data')[0].split('/')[-1]
    data_train = pd.read_excel(data_train_path)

    # CPC 1-2 --> label 0,  CPC3-5 --> label 1
    data_train['label'] = pd.cut(data_train['CPC'], bins=[0, 2, 5], labels=[1, 0])
    label0_num = len(data_train[data_train['label'] == 0])
    label1_num = len(data_train[data_train['label'] == 1])
    data_train = data_train.drop(['CPC'], axis=1)  
    print(f"训练集形状: {data_train.shape}, 训练集时间: {ct_mode}")
    result_folder = get_next_result_folder(base_path='./results_0728_delete')

    y_index_s = data_train.iloc[:, -1]
    scale_pos_weight = len(y_index_s[y_index_s == 0]) / len(y_index_s[y_index_s == 1])
    scaler = StandardScaler()
    print(f"数据划分完毕")
    best_auc = 0

    # 初始化分类器
    # clf = GaussianNB(priors=[0.83, 0.17])    # priors=[0.5, 0.5]
    # clf = LogisticRegression()

    lgbm_params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'binary_logloss',  # 使用logloss作为评价指标
        'learning_rate': 0.016,
        'max_depth': 6,
        'n_estimators': 500,
        'subsample': 0.7,  # 构建每棵树时使用的样本比例
        'colsample_bytree': 0.7,  # 每棵树使用的特征比例
        'scale_pos_weight': scale_pos_weight,  # 根据数据不平衡调整正负样本的权重
        'random_state': 42,
    }
    # clf = LGBMClassifier(**lgbm_params)

    catboost_params = {
        'iterations': 500,  # 迭代次数
        'depth': 6,  # 树的深度
        'learning_rate': 0.01,  # 学习率
        'loss_function': 'Logloss',  # 损失函数
        'eval_metric': 'AUC',  # AUC作为评价指标
        'scale_pos_weight': scale_pos_weight,  # 样本不平衡的调整
        'random_seed': 42,
        'verbose': 0  # 不输出训练过程
    }
    # clf = CatBoostClassifier(**catboost_params)

    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': ['logloss'],
        'learning_rate': 0.016,
        'max_depth': 6,
        'n_estimators': 600,
        'subsample': 0.72,         # 用于构建每棵树的样本比例
        'colsample_bytree': 0.705,  # 控制每棵树在构建时使用的特征比例
        # 'scale_pos_weight': 1.4,   # 根据实际正负样本比例设置权重 len(y[y==0]) / len(y[y==1]),
        'gamma': 0.1,
        'min_child_weight': 1,     # 降低以增加模型灵活性
        'scale_pos_weight': scale_pos_weight,
        'random_state': 4,
        # 'tree_method': 'hist',  
        # 'device': 'cuda',
    }
    # clf = xgb.XGBClassifier(**xgb_params)

    clf = SVC(
        kernel='rbf',              # 使用RBF核函数
        C=10,                     # 正则化参数
        gamma='auto',              # scale
        probability=True,          # 启用概率估计
        class_weight='balanced', # 处理类别不平衡 
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
    save_path = './results/roc_curve_{}_time_{}'.format(mode, ct_mode)
    os.makedirs(save_path, exist_ok=True)

    metrics_history = {
        'ACC': [], 'Recall': [], 'Specificity': [], 'Precision': [], 'NPV': [], 'AUC': [], 
    }
    random_states = [3, 13, 42, 87, 1307]
    results = "classifier mode is: {}\n\nrandom_states: {}\n".format(mode, random_states)
    for fold in range(1, 6):
        print(f"==============第 {fold} 次实验==============")
        X = data_train.drop('label', axis=1)
        y = data_train['label']
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X, y, 
        #     test_size=0.4,  # 40% 验证集, 76 val  113 train
        #     stratify=y,   
        #     shuffle=True,  
        #     random_state=random_states[fold-1]  
        # ) 
        X_train, X_val, y_train, y_val = balanced_train_test_split(
            X, y, 
            test_size=0.4,  # 40% 验证集, 76 val  113 train
            random_state=random_states[fold-1]
        )
        train_index = X_train.index
        lasso_train = data_train.iloc[train_index]
        results += "\ntrain index: \n{}\ntest index: \n{}\n".format(X_train.index.tolist(), X_val.index.tolist())
        results += f"train: label 0 num: {y_train.values.tolist().count(0)}, label 1 num: {y_train.values.tolist().count(1)}\n"
        results += f"test : label 0 num: {y_val.values.tolist().count(0)}, label 1 num: {y_val.values.tolist().count(1)}\n\n\n"

        # 特征降维
        print(f"data for {fold} time, train shape is : {X_train.shape}")
        data_train_lasso, selected_features, best_alphas = lasso_dimension_reduction(lasso_train)
        X_train = data_train_lasso.iloc[:, :-1] 
        y_train = data_train_lasso.iloc[:, -1]
        print(f"data train shape is : {X_train.shape}")


        print(f"data for {fold} time, selected features num is : {len(selected_features)}")
        X_val = X_val[selected_features]

        t = if_same(X_train, X_val)
        if t:
            print(f"X_train and X_val is same")
        else:
            print(f"X_train is not same as X_val")
        print(f"for {fold} time, X_test.shape: {X_val.shape}, X_train.shape: {X_train.shape}")

        # 标准化
        X_train_scaled = scaler.fit_transform(X_train)  
        X_val_scaled = scaler.transform(X_val)

        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = len(y_train) / (2 * (y_train == 0).sum())
        sample_weights[y_train == 1] = len(y_train) / (2 * (y_train == 1).sum())

        print(f"训练开始")
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)         #贝叶斯分类器 , logic分类器
        # clf.fit(X_train_scaled, y_train)   #SVM, xgb, lgbm, catboost分类器
        print(f"训练完成")

        y_pred = clf.predict(X_val_scaled)
        y_prob = clf.predict_proba(X_val_scaled)[:, -1]

        ACC, Recall, Specificity, Precision, NPV, roc_auc = calculate_metrics(y_val, y_pred, y_prob, save_roc_path=save_path, mode=mode)

        acc_scores.append(ACC)
        recall_scores.append(Recall)
        specificity_scores.append(Specificity)
        precision_scores.append(Precision)
        npv_scores.append(NPV)
        auc_scores.append(roc_auc) 
        selected_features_all.append(f"次数: {fold}, number: {len(selected_features)}, features: {selected_features}\n")

        print(f"第 {fold} 次实验ACC:{ACC:.3f}")
        print(f"第 {fold} 次实验Recall:{Recall:.3f}")
        print(f"第 {fold} 次实验Specificity:{Specificity:.3f}")
        print(f"第 {fold} 次实验Precision:{Precision:.3f}")
        print(f"第 {fold} 次实验NPV:{NPV:.3f}")
        print(f"第 {fold} 次实验AUC:{roc_auc:.3f}")
                
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
    # results = "classifier mode is: {}\n\ntrain index: \n{}\n\ntest index: \n{}".format(mode, X_train.index, X_val.index)
    results += "all dataset label 0 num: {}, label 1 num: {}\n".format(label0_num, label1_num)
    print(f"all dataset label 0 num: {label0_num}, label 1 num: {label1_num}\n")
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