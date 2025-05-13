# data.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import random
from HYPERPARAMETERS import HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS

warnings.filterwarnings('ignore')

# 2. 设置随机种子和设备
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cpu')  # 如有条件可改为 'cuda' 

# 定义全局常量
SENSITIVE_COLUMN = 'Gender'
A_PRIVILEGED = 1  # Male
A_UNPRIVILEGED = 0  # Female

# 4. 数据加载和预处理
data_train_path = r'e:\Code\FedSCOPE\code\data\depression\depression_dataset_modified.csv'

# 根据展示的前几行构造全部列名
columns = [
    'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
    'Study Satisfaction', 'Job Satisfaction', 'Suicidal_Thoughts', 
    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 
    'Depression', 'Sleep_Duration_Score', 'Dietary_Habits_Score'
]

# 选择需要使用的特征及目标变量
features_to_keep = [
    'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
    'Study Satisfaction', 'Job Satisfaction', 'Suicidal_Thoughts', 
    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 
    'Sleep_Duration_Score', 'Dietary_Habits_Score'
]
target = 'Depression'

# 读取数据（数据集带表头）
df = pd.read_csv(data_train_path)

# 保留所需特征和目标变量
df = df[features_to_keep + [target]]

# 数据清理
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# 处理敏感变量 Gender: 保证 Male→1，Female→0（这里既可能为数字也可能为字符串）
df['Gender'] = df['Gender'].astype(str).apply(lambda x: 1 if x.strip() == '1' or x.strip().lower()=='male' else 0)

# 分离特征和目标
X = df[features_to_keep]
y = df[target]

# 此数据集全为数值数据，不需要类别编码

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

print("\n===== 训练集和测试集大小 =====")
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

print("\n===== 修改前的测试集分布 =====")
print(y_test.value_counts())

# 平衡测试集
min_count = y_test.value_counts().min()
desired_class_0 = min_count
desired_class_1 = min_count
indices_class_0 = y_test[y_test == 0].index
indices_class_1 = y_test[y_test == 1].index
np.random.seed(SEED)
sampled_class_0 = np.random.choice(indices_class_0, desired_class_0, replace=False)
sampled_class_1 = np.random.choice(indices_class_1, desired_class_1, replace=False)
sampled_indices = np.concatenate([sampled_class_0, sampled_class_1])
X_test = X_test.loc[sampled_indices].reset_index(drop=True)
y_test = y_test.loc[sampled_indices].reset_index(drop=True)

print("\n===== 修改后的测试集分布 =====")
print(y_test.value_counts())

# 标准化连续变量（不对 Gender 标准化）
numerical_columns = [col for col in features_to_keep if col != 'Gender']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# 转换数据类型为 float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(DEVICE)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(DEVICE)

# 提取敏感特征 Gender（假设在第0列）
gender_test_tensor = X_test_tensor[:, 0].long().to(DEVICE)

# 创建测试数据加载器（包含敏感特征）
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor, gender_test_tensor),
    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
    shuffle=False
)

print(f"\n修改后测试集大小: {X_test_tensor.shape}")
print(f"训练集大小: {X_train_tensor.shape}")
print(f"测试集大小: {X_test_tensor.shape}")
print(f"训练目标变量大小: {y_train_tensor.shape}")
print(f"测试目标变量大小: {y_test_tensor.shape}")

# 5. 使用 Dirichlet 分布生成非IID数据分布（同时分出一个 10% 的服务器数据集）

def split_server_client_data(train_df, HYPERPARAMETERS):
    """
    将原数据集中的 10% 取出作为服务器的数据集（IID），剩余 90% 用于客户端的非IID分布。
    """
    total_train_size = len(train_df)
    server_size_total = max(1, int(0.1 * total_train_size))
    server_privileged_ratio = train_df[SENSITIVE_COLUMN].mean()
    server_privileged_size = int(server_privileged_ratio * server_size_total)
    server_unprivileged_size = server_size_total - server_privileged_size
    server_privileged_size = min(server_privileged_size, len(train_df[train_df[SENSITIVE_COLUMN] == A_PRIVILEGED]))
    server_unprivileged_size = min(server_unprivileged_size, len(train_df[train_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED]))
    server_privileged = train_df[train_df[SENSITIVE_COLUMN] == A_PRIVILEGED].sample(n=server_privileged_size, random_state=SEED)
    server_unprivileged = train_df[train_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED].sample(n=server_unprivileged_size, random_state=SEED)
    server_df = pd.concat([server_privileged, server_unprivileged])
    print("\n===== Server Data =====")
    print(f"Server training dataset size: {len(server_df)}")
    print(f"Male Count: {np.sum(server_df[SENSITIVE_COLUMN] == A_PRIVILEGED)}, Female Count: {np.sum(server_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED)}")
    client_df = train_df.drop(server_df.index).reset_index(drop=True)
    return server_df.reset_index(drop=True), client_df

server_df, client_df = split_server_client_data(X_train.assign(**{target: y_train}), HYPERPARAMETERS)

# 利用 Dirichlet 分布划分客户端数据（计算方式与原代码相同，只是字段名称作了更新）
ALPHA = HYPERPARAMETERS['ALPHA_DIRICHLET'][0]
NUM_CLIENTS = HYPERPARAMETERS['NUM_CLIENTS']
client_df.reset_index(drop=True, inplace=True)
X_train_df = client_df[features_to_keep]
y_train_df = client_df[target]
privileged_indices = client_df[client_df[SENSITIVE_COLUMN] == A_PRIVILEGED].index
unprivileged_indices = client_df[client_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED].index
privileged_X = X_train_df.loc[privileged_indices].reset_index(drop=True)
privileged_y = y_train_df.loc[privileged_indices].reset_index(drop=True)
unprivileged_X = X_train_df.loc[unprivileged_indices].reset_index(drop=True)
unprivileged_y = y_train_df.loc[unprivileged_indices].reset_index(drop=True)
privileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
unprivileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
client_data_dict = {i: {"X": [], "y": [], "sensitive": []} for i in range(NUM_CLIENTS)}

privileged_splits = (privileged_ratios * len(privileged_X)).astype(int)
start_idx = 0
for i, count in enumerate(privileged_splits):
    end_idx = start_idx + count
    if end_idx > len(privileged_X):
        end_idx = len(privileged_X)
    client_data_dict[i]["X"].append(privileged_X.iloc[start_idx:end_idx].values)
    client_data_dict[i]["y"].append(privileged_y.iloc[start_idx:end_idx].values)
    client_data_dict[i]["sensitive"].append(np.full(end_idx - start_idx, A_PRIVILEGED))
    start_idx = end_idx

remaining = len(privileged_X) - start_idx
if remaining > 0:
    client_data_dict[NUM_CLIENTS - 1]["X"].append(privileged_X.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["y"].append(privileged_y.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_PRIVILEGED))

unprivileged_splits = (unprivileged_ratios * len(unprivileged_X)).astype(int)
start_idx = 0
for i, count in enumerate(unprivileged_splits):
    end_idx = start_idx + count
    if end_idx > len(unprivileged_X):
        end_idx = len(unprivileged_X)
    client_data_dict[i]["X"].append(unprivileged_X.iloc[start_idx:end_idx].values)
    client_data_dict[i]["y"].append(unprivileged_y.iloc[start_idx:end_idx].values)
    client_data_dict[i]["sensitive"].append(np.full(end_idx - start_idx, A_UNPRIVILEGED))
    start_idx = end_idx

remaining = len(unprivileged_X) - start_idx
if remaining > 0:
    client_data_dict[NUM_CLIENTS - 1]["X"].append(unprivileged_X.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["y"].append(unprivileged_y.iloc[start_idx:].values)
    client_data_dict[NUM_CLIENTS - 1]["sensitive"].append(np.full(remaining, A_UNPRIVILEGED))

for i in range(NUM_CLIENTS):
    if len(client_data_dict[i]["X"]) > 0:
        client_data_dict[i]["X"] = torch.tensor(np.vstack(client_data_dict[i]["X"]), dtype=torch.float32).to(DEVICE)
    else:
        client_data_dict[i]["X"] = torch.empty(0, len(features_to_keep), dtype=torch.float32).to(DEVICE)
    if len(client_data_dict[i]["y"]) > 0:
        client_data_dict[i]["y"] = torch.tensor(np.concatenate(client_data_dict[i]["y"]), dtype=torch.long).to(DEVICE)
    else:
        client_data_dict[i]["y"] = torch.empty(0, dtype=torch.long).to(DEVICE)
    if len(client_data_dict[i]["sensitive"]) > 0:
        client_data_dict[i]["sensitive"] = np.concatenate(client_data_dict[i]["sensitive"]).astype(int)
    else:
        client_data_dict[i]["sensitive"] = np.array([], dtype=int)

stats = []
for i in range(NUM_CLIENTS):
    privileged_count = np.sum(client_data_dict[i]["sensitive"] == A_PRIVILEGED)
    unprivileged_count = np.sum(client_data_dict[i]["sensitive"] == A_UNPRIVILEGED)
    stats.append([i, privileged_count, unprivileged_count])
stats_df = pd.DataFrame(stats, columns=["客户端", "Male 数量", "Female 数量"])
print("\n每个客户端中 Gender 的数量分布：")
print(stats_df)
for i in range(NUM_CLIENTS):
    total_count = len(client_data_dict[i]["X"])
    expected_count = stats_df.iloc[i]["Male 数量"] + stats_df.iloc[i]["Female 数量"]
    print(f"\n客户端 {i} 的训练数据集大小: {total_count} (Male + Female)\n")
    assert total_count == expected_count, f"客户端 {i} 的训练数据与统计数据不匹配！"

print(f"训练集大小: {X_train_tensor.shape}")
print(f"测试集大小: {X_test_tensor.shape}")
print(f"训练目标变量大小: {y_train_tensor.shape}")
print(f"测试目标变量大小: {y_test_tensor.shape}")