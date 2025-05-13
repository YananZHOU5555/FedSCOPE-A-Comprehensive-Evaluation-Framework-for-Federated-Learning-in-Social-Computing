# data.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
SENSITIVE_COLUMN = 'race'
A_PRIVILEGED = 1  # African-American
A_UNPRIVILEGED = 0  # 非 African-American

# 4. 数据加载和预处理
data_train_path = r'e:\Code\FedSCOPE\code\data\compas\compas-scores-two-years.csv'
# 注意：COMPAS 数据集的 train 和 test 合在一个文件中

# 根据数据集中各列说明定义列名
columns = [
    'id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob', 'age', 'age_cat',
    'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count',
    'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number', 'c_offense_date',
    'c_arrest_date', 'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid',
    'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 'r_offense_date',
    'r_charge_desc', 'r_jail_in', 'r_jail_out', 'violent_recid', 'is_violent_recid',
    'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
    'type_of_assessment', 'decile_score_duplicate', 'score_text', 'screening_date',
    'v_type_of_assessment', 'v_decile_score', 'v_score_text', 'v_screening_date',
    'in_custody', 'out_custody', 'priors_count_duplicate', 'start', 'end', 'event',
    'two_year_recid'
]

features_to_keep = [
    'sex', 'age', 'age_cat', 'race',
    'juv_fel_count', 'juv_misd_count', 'juv_other_count',
    'priors_count', 'c_charge_degree', 'c_charge_desc'
]
target = 'two_year_recid'

# 读取数据
df = pd.read_csv(data_train_path, header=0, names=columns, skipinitialspace=True)

# 保留所需特征和目标变量
df = df[features_to_keep + [target]]

# 数据清理
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# 处理敏感变量 'race': "African-American" 为 1，其余为 0
df['race'] = df['race'].apply(lambda x: 1 if x.strip().lower() == 'african-american' else 0)

# 分离特征和目标
X = df[features_to_keep]
y = df[target]

# 对类别变量进行编码
categorical_columns = ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 分割训练集和测试集（保持 stratify）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

print("\n===== 训练集和测试集大小 =====")
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

print("\n===== 修改前的测试集分布 =====")
print(y_test.value_counts())

# 平衡测试集（确保每个类别的样本数相同）
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

# 标准化连续变量
numerical_columns = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# 转换为浮点型
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(DEVICE)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(DEVICE)

# 提取敏感特征 'race'，假设在特征中的第4列（索引3）
race_test_tensor = X_test_tensor[:, 3].long().to(DEVICE)

# 创建包含敏感属性 race 的测试数据加载器
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor, race_test_tensor),
    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
    shuffle=False
)

print(f"\n修改后测试集大小: {X_test_tensor.shape}")
print(f"训练集大小: {X_train_tensor.shape}")
print(f"测试集大小: {X_test_tensor.shape}")
print(f"训练目标变量大小: {y_train_tensor.shape}")
print(f"测试目标变量大小: {y_test_tensor.shape}")

# 5. 使用 Dirichlet 分布生成非IID数据分布（并划分出10%的服务器数据集）
def split_server_client_data(train_df, HYPERPARAMETERS):
    """
    将原数据集中的10%取出作为服务器的数据集（保持种族分布 IID），剩下90%作为客户端数据。
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
    print(f"African-American Count: {np.sum(server_df[SENSITIVE_COLUMN] == A_PRIVILEGED)}, Others Count: {np.sum(server_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED)}")
    client_df = train_df.drop(server_df.index).reset_index(drop=True)
    return server_df.reset_index(drop=True), client_df

server_df, client_df = split_server_client_data(X_train.assign(two_year_recid=y_train), HYPERPARAMETERS)

# 使用 Dirichlet 分布生成客户端数据（非IID 分布）
ALPHA = HYPERPARAMETERS['ALPHA_DIRICHLET'][0]
NUM_CLIENTS = HYPERPARAMETERS['NUM_CLIENTS']
client_df.reset_index(drop=True, inplace=True)
X_train_df = client_df[features_to_keep]
y_train_df = client_df[target]

# 分离特权和非特权群体
privileged_indices = client_df[client_df[SENSITIVE_COLUMN] == A_PRIVILEGED].index
unprivileged_indices = client_df[client_df[SENSITIVE_COLUMN] == A_UNPRIVILEGED].index
privileged_X = X_train_df.loc[privileged_indices].reset_index(drop=True)
privileged_y = y_train_df.loc[privileged_indices].reset_index(drop=True)
unprivileged_X = X_train_df.loc[unprivileged_indices].reset_index(drop=True)
unprivileged_y = y_train_df.loc[unprivileged_indices].reset_index(drop=True)

# Dirichlet 分配比例
privileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
unprivileged_ratios = np.random.dirichlet([ALPHA] * NUM_CLIENTS, size=1)[0]
client_data_dict = {i: {"X": [], "y": [], "sensitive": []} for i in range(NUM_CLIENTS)}

# 分配特权群体数据（African-American）
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

# 分配非特权群体数据（非 African-American）
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
stats_df = pd.DataFrame(stats, columns=["客户端", "African-American 数量", "Others 数量"])
print("\n每个客户端中种族的数量分布：")
print(stats_df)
for i in range(NUM_CLIENTS):
    total_count = len(client_data_dict[i]["X"])
    expected_count = stats_df.iloc[i]["African-American 数量"] + stats_df.iloc[i]["Others 数量"]
    print(f"\n客户端 {i} 的训练数据集大小: {total_count} (African-American + Others)\n")
    assert total_count == expected_count, f"客户端 {i} 的训练数据与统计数据不匹配！"

print(f"训练集大小: {X_train_tensor.shape}")
print(f"测试集大小: {X_test_tensor.shape}")
print(f"训练目标变量大小: {y_train_tensor.shape}")
print(f"测试目标变量大小: {y_test_tensor.shape}")