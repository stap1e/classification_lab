# CPC 预后分类（影像组学 + NSE）

基于 CT 影像组学特征与 NSE 实验室指标，对心脏骤停患者 CPC 评分进行二分类（CPC 1–2 vs 3–5），支持多种机器学习模型与 LASSO 特征筛选。

## 仓库结构

```
code/cls/
├── config/           # 实验默认配置（标签分箱、列名等）
├── models/           # 分类器工厂（统一超参入口）
├── pipeline/         # 数据准备 + 训练评估主流程
├── utils/            # 工具函数（LASSO、指标、结果保存）
│   ├── pre4data.py
│   ├── util.py
│   └── tools/        # 数据对齐、NSE 处理等脚本
├── classify_lab1.py  # 固定 train/test + 可选 NSE（lab1）
├── classify_single.py# 单文件划分 train/test
├── classify_kfold.py # 5 折交叉验证
├── feature_process.py# 特征表合并与清洗（数据预处理）
├── new_data_process.py
├── train_test.py     # 划分并导出 train/test Excel
└── requirements.txt
```

## 环境

```bash
cd code/cls
pip install -r requirements.txt
```

## 运行实验

在 `code/cls` 目录下执行（保证 `config`、`pipeline` 可被导入）：

**Lab1：固定训练/测试集，可选 NSE**

```bash
python classify_lab1.py
```

在脚本内修改 `ExperimentConfig` 中的 `train_path`、`test_path`、`results_base_dir`、`use_nse`、`classifier`。

**单表随机划分**

```bash
python classify_single.py
```

**5 折交叉验证**

```bash
python classify_kfold.py
```

### 可选分类器

`classifier` 取值：`svm` | `logistic` | `gaussian_nb` | `xgboost` | `lightgbm` | `catboost`

### 标签定义

默认：CPC 1–2 → 0，CPC 3–5 → 1。可在 `ExperimentConfig` 中调整 `cpc_bins` / `cpc_labels`。

## 设计说明

- **配置与逻辑分离**：路径、模型、是否使用 NSE 等集中在 `ExperimentConfig`，避免在多个脚本中复制数百行训练代码。
- **统一流水线**：`pipeline.experiment.run_experiment` 负责 LASSO、标准化、训练、指标汇总与结果写入。
- **可扩展**：新增模型只需在 `models/factory.py` 注册；新增实验类型可复用 `pipeline.data` 与 `_run_single_fold`。

## 数据预处理脚本

| 脚本 | 用途 |
|------|------|
| `feature_process.py` | 多时间点影像组学表合并、剔除 diagnostics 列 |
| `feature_process.py` / `new_data_process.py` | 院内数据流水线（需配置本地 Excel 路径） |
| `utils/tools/*` | CTid–姓名–NSE 对齐等 |

路径默认为历史 Windows 绝对路径，请按本机数据位置修改各脚本顶部的路径常量，或后续改为环境变量 / YAML 配置。
