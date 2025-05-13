# data.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import matplotlib as mpl
# 导入绘图所需库
import matplotlib.pyplot as plt
import seaborn as sns
from HYPERPARAMETERS import HYPERPARAMETERS,SEED,algorithms,attack_forms,MALICIOUS_CLIENTS  #导入HYPERPARAMETERS




warnings.filterwarnings('ignore')

# 2. 设置随机种子和设备
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cpu')


# 4. 数据加载和预处理
data_train_path = 'e:\Code\FedSCOPE\code\data/adult/adult.data'
data_test_path = 'e:\Code\FedSCOPE\code\data/adult/adult.data'

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country', 'income'
]

train_df = pd.read_csv(data_train_path, header=None, names=columns, skipinitialspace=True)
test_df = pd.read_csv(data_test_path, header=None, names=columns, skipinitialspace=True, skiprows=1)

# 数据清理
train_df.replace('?', np.nan, inplace=True)
test_df.replace('?', np.nan, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

# 类别变量编码
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country', 'income'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    train_classes = le.classes_
    if not all(test_df[col].isin(train_classes)):
        test_df[col] = test_df[col].apply(lambda x: 'Unknown' if x not in train_classes else x)
        if 'Unknown' not in train_classes:
            le_classes = np.append(le.classes_, 'Unknown')
            le.classes_ = le_classes
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# 分离特征和目标
X_train = train_df.drop('income', axis=1)
y_train = train_df['income']

X_test = test_df.drop('income', axis=1)
y_test = test_df['income']
sex_test = test_df['sex']  # 提取性别信息

# 标准化连续变量
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
sex_test_tensor = torch.tensor(sex_test.values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])

# 创建包含sex的测试数据加载器
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor, sex_test_tensor),
    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
    shuffle=False
)

print(f"修改后测试集大小: {X_test_tensor.shape}")
print(f"训练集大小: {X_train_tensor.shape}")
print(f"测试集大小: {X_test_tensor.shape}")
print(f"训练目标变量大小: {y_train_tensor.shape}")
print(f"测试目标变量大小: {y_test_tensor.shape}")

# 5. 使用 Dirichlet 分布生成非IID数据分布
def split_server_client_data(train_df, HYPERPARAMETERS):
    """
    将原数据集中的10%取出作为服务器的数据集，保证与原数据的性别分布一致（IID）。
    剩下的90%用于客户端的非IID分布。
    """
    SENSITIVE_COLUMN = 'sex'
    A_PRIVILEGED = 1  # 男性
    A_UNPRIVILEGED = 0  # 女性

    # 计算原始数据的性别分布
    total_train_size = len(train_df)
    server_size_total = max(1, int(0.1 * total_train_size))  # 至少取1个样本

    # 计算服务器数据集中男性和女性的数量，保持与原始数据相同的比例
    server_male_ratio = train_df[SENSITIVE_COLUMN].mean()
    server_male_size = int(server_male_ratio * server_size_total)
    server_female_size = server_size_total - server_male_size

    # 确保不会超过实际数量
    server_male_size = min(server_male_size, len(train_df[train_df[SENSITIVE_COLUMN] == A_PRIVILEGED]))
    server_female_size = min(server_female_size, len(train_df[train_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED]))

    # 随机选择服务器数据
    server_privileged = train_df[train_df[SENSITIVE_COLUMN] == A_PRIVILEGED].sample(n=server_male_size, random_state=SEED)
    server_unprivileged = train_df[train_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED].sample(n=server_female_size, random_state=SEED)
    server_df = pd.concat([server_privileged, server_unprivileged])

    # 打印服务器数据集大小及性别分布
    server_male_count = np.sum(server_df[SENSITIVE_COLUMN] == A_PRIVILEGED)
    server_female_count = np.sum(server_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED)
    print("\n===== Server Data =====")
    print(f"Server training dataset size: {len(server_df)}")
    print(f"Male Count: {server_male_count}, Female Count: {server_female_count}")

    # 剩余数据用于客户端
    client_df = train_df.drop(server_df.index).reset_index(drop=True)

    return server_df.reset_index(drop=True), client_df

# 分离服务器和客户端数据
server_df, client_df = split_server_client_data(train_df, HYPERPARAMETERS)

# 使用 Dirichlet 分布生成非IID数据分布
ALPHA = HYPERPARAMETERS['ALPHA_DIRICHLET'][0]  # Dirichlet 分布的浓度参数
NUM_CLIENTS = HYPERPARAMETERS['NUM_CLIENTS']  # 客户端数量
SENSITIVE_COLUMN = 'sex'  # 敏感属性列（性别）
A_PRIVILEGED = 1  # 特权群体（男性）的编码
A_UNPRIVILEGED = 0  # 非特权群体（女性）的编码

# 确保 X_train 和 y_train 的索引与 train_df 保持一致
train_df.reset_index(drop=True, inplace=True)
X_train_df = pd.DataFrame(X_train_tensor.cpu().numpy(), columns=train_df.drop('income', axis=1).columns, index=train_df.index)
y_train_df = pd.Series(y_train_tensor.cpu().numpy(), index=train_df.index)

# 分离特权群体和非特权群体 (仍为 DataFrame)
privileged_indices = train_df[train_df[SENSITIVE_COLUMN] == A_PRIVILEGED].index
unprivileged_indices = train_df[train_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED].index

privileged_X = X_train_df.loc[privileged_indices].reset_index(drop=True)
privileged_y = y_train_df.loc[privileged_indices].reset_index(drop=True)

unprivileged_X = X_train_df.loc[unprivileged_indices].reset_index(drop=True)
unprivileged_y = y_train_df.loc[unprivileged_indices].reset_index(drop=True)

# Dirichlet 分配比例
privileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
unprivileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]

# 初始化客户端数据
client_data_dict = {i: {"X": [], "y": [], "sensitive": []} for i in range(NUM_CLIENTS)}

# 分配特权群体数据（男性）
privileged_splits = (privileged_ratios * len(privileged_X)).astype(int)
start_idx = 0
for i, count in enumerate(privileged_splits):
    end_idx = start_idx + count
    if end_idx > len(privileged_X):
        end_idx = len(privileged_X)
    client_data_dict[i]["X"].append(privileged_X.iloc[start_idx:end_idx].values)
    client_data_dict[i]["y"].append(privileged_y.iloc[start_idx:end_idx].values)
    client_data_dict[i]["sensitive"].append(np.full(end_idx - start_idx, A_PRIVILEGED))  # 记录敏感属性（男性）
    start_idx = end_idx

# 处理剩余的样本（如果有）
remaining = len(privileged_X) - start_idx
if remaining > 0:
    client_data_dict[NUM_CLIENTS - 1]["X"].append(privileged_X.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["y"].append(privileged_y.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_PRIVILEGED))

# 分配非特权群体数据（女性）
unprivileged_splits = (unprivileged_ratios * len(unprivileged_X)).astype(int)
start_idx = 0
for i, count in enumerate(unprivileged_splits):
    end_idx = start_idx + count
    if end_idx > len(unprivileged_X):
        end_idx = len(unprivileged_X)
    client_data_dict[i]["X"].append(unprivileged_X.iloc[start_idx:end_idx].values)
    client_data_dict[i]["y"].append(unprivileged_y.iloc[start_idx:end_idx].values)
    client_data_dict[i]["sensitive"].append(np.full(end_idx - start_idx, A_UNPRIVILEGED))  # 记录敏感属性（女性）
    start_idx = end_idx

# 处理剩余的样本（如果有）
remaining = len(unprivileged_X) - start_idx
if remaining > 0:
    client_data_dict[NUM_CLIENTS - 1]["X"].append(unprivileged_X.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["y"].append(unprivileged_y.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_UNPRIVILEGED))

# 合并特权和非特权群体数据，并转换为 Tensor
for i in range(NUM_CLIENTS):
    if len(client_data_dict[i]["X"]) > 0:
        client_data_dict[i]["X"] = torch.tensor(np.vstack(client_data_dict[i]["X"]), dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
    else:
        client_data_dict[i]["X"] = torch.empty(0, X_train_tensor.shape[1], dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])

    if len(client_data_dict[i]["y"]) > 0:
        client_data_dict[i]["y"] = torch.tensor(np.concatenate(client_data_dict[i]["y"]), dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
    else:
        client_data_dict[i]["y"] = torch.empty(0, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])

    if len(client_data_dict[i]["sensitive"]) > 0:
        client_data_dict[i]["sensitive"] = np.concatenate(client_data_dict[i]["sensitive"]).astype(int)
    else:
        client_data_dict[i]["sensitive"] = np.array([], dtype=int)

# 统计每个客户端中男性和女性的数量
stats = []
for i in range(NUM_CLIENTS):
    male_count = np.sum(client_data_dict[i]["sensitive"] == A_PRIVILEGED)
    female_count = np.sum(client_data_dict[i]["sensitive"] == A_UNPRIVILEGED)
    stats.append([i, male_count, female_count])

# 创建一个表格展示统计信息
stats_df = pd.DataFrame(stats, columns=["客户端", "男性数量", "女性数量"])
print("\n每个客户端中男性和女性的数量分布：")
print(stats_df)

# 验证：每个客户端的数据就是其训练数据
for i in range(NUM_CLIENTS):
    total_count = len(client_data_dict[i]["X"])
    expected_count = stats_df.iloc[i]["男性数量"] + stats_df.iloc[i]["女性数量"]
    print(f"\n客户端 {i} 的训练数据集大小: {total_count} (男性 + 女性)\n")
    assert total_count == expected_count, f"客户端 {i} 的训练数据与统计数据不匹配！"