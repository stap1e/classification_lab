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
├── classify_kfold.py # 5 折交叉验证（无 NSE）
├── classify_kfold_nse.py # 5 折 + NSE
├── run_experiment.py # 统一 CLI（任意 profile）
├── config/experiments.yaml # 路径与实验配置（外置）
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

## 配置路径（YAML + 环境变量）

所有实验参数集中在 `code/cls/config/experiments.yaml`。路径使用占位符，无需改 Python 代码：

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `CLS_DATA_ROOT` | Excel 数据目录 | `./data` |
| `CLS_RESULTS_ROOT` | 结果输出根目录 | `./results` |
| `CLS_EXPERIMENT` | 默认 profile | 各入口脚本不同 |

YAML 内可使用 `{data_root}/文件名.xlsx`，或 `${CLS_DATA_ROOT:-./data}/文件名.xlsx`。

**Windows 示例**

```bat
set CLS_DATA_ROOT=D:\thrid_beijing_hospital_data
set CLS_RESULTS_ROOT=D:\thrid_beijing_hospital_data
python classify_lab1.py
```

**Linux / macOS 示例**

```bash
export CLS_DATA_ROOT=/path/to/your/data
export CLS_RESULTS_ROOT=/path/to/your/results
python run_experiment.py -p kfold_nse
```

查看可用 profile：

```bash
python run_experiment.py --list-profiles
```

| Profile | 说明 |
|---------|------|
| `lab1` | 固定 train/test，含 NSE |
| `single` | 单表 8:2 分层划分 |
| `kfold` | 5 折交叉验证（仅 CT 特征） |
| `kfold_nse` | 5 折 + NSE（每折独立 LASSO 与 NSE 拼接） |

修改或新增实验：编辑 `config/experiments.yaml` 中 `experiments` 节点，或复制一份自定义 YAML 并用 `-c` 指定。

## 运行实验

在 `code/cls` 目录下执行：

```bash
pip install -r requirements.txt
python classify_lab1.py          # 等同: python run_experiment.py -p lab1
python classify_single.py
python classify_kfold.py
python classify_kfold_nse.py       # 5 折 + NSE
python run_experiment.py -p lab1 -c config/experiments.yaml
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

预处理脚本（`feature_process.py`、`utils/tools/*`）仍可能含历史绝对路径；建议同样改为读取 `CLS_DATA_ROOT` 或从 `config/experiments.yaml` 的 `paths` 段复制路径约定。
