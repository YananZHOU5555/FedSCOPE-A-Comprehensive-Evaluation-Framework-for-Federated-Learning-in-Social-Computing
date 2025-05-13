# function.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # Ensure f1_score is imported
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
from HYPERPARAMETERS import DEVICE,HYPERPARAMETERS,SEED,algorithms,attack_forms,MALICIOUS_CLIENTS  #导入HYPERPARAMETERS
from data import A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN,test_loader,client_data_dict,X_test_tensor,y_test_tensor,sex_test_tensor,X_train_tensor,y_train_tensor,X_test,y_test,sex_test,scaler,categorical_columns,label_encoders,numerical_columns,train_df,test_df,X_train,y_train,X_test,y_test,sex_test,test_loader,A_PRIVILEGED,A_UNPRIVILEGED,algorithms,attack_forms,MALICIOUS_CLIENTS 

# 6. 公平性指标计算
def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    # 定义优势群体（男性）和非优势群体（女性）
    privileged_group = (sensitive_features == A_PRIVILEGED)  # 优势群体（男性）
    unprivileged_group = (sensitive_features == A_UNPRIVILEGED)  # 非优势群体（女性）

    ###### EOD 计算##########
    TPR_privileged = (
        np.sum((y_pred == 1) & (y_true == 1) & privileged_group) / np.sum((y_true == 1) & privileged_group)
        if np.sum((y_true == 1) & privileged_group) > 0 else 0)
    TPR_unprivileged = (
        np.sum((y_pred == 1) & (y_true == 1) & unprivileged_group) / np.sum((y_true == 1) & unprivileged_group)
        if np.sum((y_true == 1) & unprivileged_group) > 0 else 0)
    EOD = TPR_unprivileged -  TPR_privileged

    ###### SPD 计算##########
    SPD_privileged = (
        np.sum((y_pred == 1)  & privileged_group) / np.sum(privileged_group)
        if np.sum(privileged_group) > 0 else 0)
    SPD_unprivileged = (
        np.sum((y_pred == 1) & unprivileged_group) / np.sum(unprivileged_group)
        if np.sum(unprivileged_group) > 0 else 0)
    SPD =  SPD_unprivileged - SPD_privileged

    return {"SPD": SPD,  "EOD": EOD,  "TPR_privileged": TPR_privileged,  "TPR_unprivileged": TPR_unprivileged}
        

# 9. 定义测试函数并修改以返回每个类别的统计信息
def test_inference_modified(global_model, test_loader, model_class):
    """
    测试全局模型的性能，并计算每个类别的总数和正确预测数。
    """
    global_model.eval()
    y_true, y_pred, y_prob, sex_values = [], [], [], []
    loss_total = 0.0
    criteria = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X_batch, y_batch, sex_batch = batch
            else:
                X_batch, y_batch = batch
                sex_batch = torch.tensor([]).to(HYPERPARAMETERS['DEVICE'])  # 空张量

            X_batch, y_batch, sex_batch = (
                X_batch.to(HYPERPARAMETERS['DEVICE']),
                y_batch.to(HYPERPARAMETERS['DEVICE']),
                sex_batch.to(HYPERPARAMETERS['DEVICE'])
            )
            logits = global_model(X_batch)
            loss = criteria(logits, y_batch).item()
            loss_total += loss * X_batch.size(0)
            probs = nn.functional.softmax(logits, dim=1)[:, 1]  # 类别1的概率
            preds = torch.argmax(logits, dim=1)
            y_pred_batch = preds.cpu().numpy()
            y_true_batch = y_batch.cpu().numpy()
            y_prob_batch = probs.cpu().numpy()
            sex_batch = sex_batch.cpu().numpy()

            y_pred.extend(y_pred_batch)
            y_true.extend(y_true_batch)
            y_prob.extend(y_prob_batch)
            sex_values.extend(sex_batch)

    loss_test = loss_total / len(y_true) if len(y_true) > 0 else 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    sex_values = np.array(sex_values)

    # 计算公平性指标
    fairness_metrics = compute_fairness_metrics(y_true, y_pred, sex_values)

    # 计算 F1 Score
    f1 = f1_score(y_true, y_pred, average='binary')

    # 计算每个类别的总数和正确预测数
    per_category = {
        '(X+,Y+)': {'total': 0, 'correct': 0},
        '(X+,Y-)': {'total': 0, 'correct': 0},
        '(X-,Y+)': {'total': 0, 'correct': 0},
        '(X-,Y-)': {'total': 0, 'correct': 0}
    }

    # 获取sex和income的真实值
    # sex_values 和 y_true 已经对齐
    for i in range(len(y_true)):
        sex = sex_values[i]
        income = y_true[i]
        prediction = y_pred[i]

        if sex == 1 and income == 1:
            per_category['(X+,Y+)']['total'] += 1
            if prediction == 1:
                per_category['(X+,Y+)']['correct'] += 1
        elif sex == 1 and income == 0:
            per_category['(X+,Y-)']['total'] += 1
            if prediction == 0:
                per_category['(X+,Y-)']['correct'] += 1
        elif sex == 0 and income == 1:
            per_category['(X-,Y+)']['total'] += 1
            if prediction == 1:
                per_category['(X-,Y+)']['correct'] += 1
        elif sex == 0 and income == 0:
            per_category['(X-,Y-)']['total'] += 1
            if prediction == 0:
                per_category['(X-,Y-)']['correct'] += 1
    # 添加验证步骤
    assert len(y_true) == len(sex_values), "y_true 和 sex_values 的长度不匹配！"

    return loss_test, accuracy_score(y_true, y_pred), f1, fairness_metrics, per_category

# 10. 定义重加权函数
def compute_reweighing_weights(train_df, sensitive_column, class_column):
    """
    计算全局重加权权重。
    """
    total = len(train_df)
    P_S = train_df[sensitive_column].value_counts(normalize=True).to_dict()
    P_C = train_df[class_column].value_counts(normalize=True).to_dict()
    # 计算Pexp(s, c) = P(S=s) * P(Class=c)
    P_exp = {}
    for s in train_df[sensitive_column].unique():
        for c in train_df[class_column].unique():
            P_exp[(s, c)] = P_S.get(s, 0) * P_C.get(c, 0)
    # 计算Pobs(s, c) = P(S=s ∧ Class=c)
    P_obs = {}
    for (s, c), count in train_df.groupby([sensitive_column, class_column]).size().items():
        P_obs[(s, c)] = count / total

    # 计算权重W(s, c) = Pexp(s, c) / Pobs(s, c)
    weights = {}
    for key in P_exp:
        obs = P_obs.get(key, 0)
        if obs == 0:
            weights[key] = 1.0  # 如果Pobs(s, c)为0，设置权重为1
        else:
            weights[key] = P_exp[key] / P_obs[key]
    return weights

# 11. 将样本权重分配给客户端
def assign_sample_weights_to_clients(clients_data, weights, sensitive_column, class_column='income'):
    """
    将样本权重分配给每个客户端的数据字典。
    """
    for client_id, client_data in clients_data.items():
        s = client_data['sensitive']
        c = client_data['y'].cpu().numpy()
        client_weights = np.array([weights.get((s_i, c_i), 1.0) for s_i, c_i in zip(s, c)])
        # 如果权重存在，则分配；否则默认权重为1.0
        clients_data[client_id]["sample_weights"] = client_weights


# 13. 定义MLP模型
class MLP(nn.Module):
    def __init__(self, num_features, num_classes, seed=123):
        torch.manual_seed(seed)
        super().__init__()
        self.linear1 = nn.Linear(num_features, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits  # CrossEntropyLoss已包含Softmax