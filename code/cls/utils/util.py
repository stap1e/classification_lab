from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
from datetime import datetime


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


def save_results(results_text, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    
    result_file = os.path.join(result_folder, 'results.txt')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_with_timestamp = f"实验时间: {timestamp}\n\n{results_text}"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(results_with_timestamp)
    
    print(f"\n结果已保存到: {result_file}")


def get_next_result_folder(base_path='D:/PycharmProject/classification/results_default'):
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