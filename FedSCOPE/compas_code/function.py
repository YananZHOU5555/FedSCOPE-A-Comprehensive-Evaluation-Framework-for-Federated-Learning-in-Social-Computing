# function.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import matplotlib as mpl
from HYPERPARAMETERS import DEVICE, HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
from data import (
    A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN, test_loader, client_data_dict,
    X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor, X_test, y_test,
    scaler, numerical_columns, HYPERPARAMETERS
)
import torch.nn.functional as F

# 6. 公平性指标计算（使用 race 作为敏感变量）
def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    privileged_group = (sensitive_features == A_PRIVILEGED)  # African-American
    unprivileged_group = (sensitive_features == A_UNPRIVILEGED)  # Others
    TPR_privileged = (
        np.sum((y_pred == 1) & (y_true == 1) & privileged_group) / np.sum((y_true == 1) & privileged_group)
        if np.sum((y_true == 1) & privileged_group) > 0 else 0
    )
    TPR_unprivileged = (
        np.sum((y_pred == 1) & (y_true == 1) & unprivileged_group) / np.sum((y_true == 1) & unprivileged_group)
        if np.sum((y_true == 1) & unprivileged_group) > 0 else 0
    )
    EOD = TPR_unprivileged - TPR_privileged
    SPD_privileged = (
        np.sum((y_pred == 1) & privileged_group) / np.sum(privileged_group)
        if np.sum(privileged_group) > 0 else 0
    )
    SPD_unprivileged = (
        np.sum((y_pred == 1) & unprivileged_group) / np.sum(unprivileged_group)
        if np.sum(unprivileged_group) > 0 else 0
    )
    SPD = SPD_unprivileged - SPD_privileged
    return {"SPD": SPD, "EOD": EOD, "TPR_privileged": TPR_privileged, "TPR_unprivileged": TPR_unprivileged}

def test_inference_modified(global_model, test_loader, model_class):
    global_model.eval()
    y_true, y_pred, y_prob, sensitive_vals = [], [], [], []
    loss_total = 0.0
    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X_batch, y_batch, sensitive_batch = batch
            else:
                X_batch, y_batch = batch
                sensitive_batch = torch.tensor([], dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
            X_batch, y_batch, sensitive_batch = (
                X_batch.to(HYPERPARAMETERS['DEVICE']),
                y_batch.to(HYPERPARAMETERS['DEVICE']),
                sensitive_batch.to(HYPERPARAMETERS['DEVICE'])
            )
            logits = global_model(X_batch)
            loss = criteria(logits, y_batch).item()
            loss_total += loss * X_batch.size(0)
            probs = nn.functional.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_pred_batch = preds.cpu().numpy()
            y_true_batch = y_batch.cpu().numpy()
            y_prob_batch = probs.cpu().numpy()
            sensitive_batch = sensitive_batch.cpu().numpy()
            y_pred.extend(y_pred_batch)
            y_true.extend(y_true_batch)
            y_prob.extend(y_prob_batch)
            sensitive_vals.extend(sensitive_batch)
    loss_test = loss_total / len(y_true) if len(y_true) > 0 else 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    sensitive_vals = np.array(sensitive_vals)
    fairness_metrics = compute_fairness_metrics(y_true, y_pred, sensitive_vals)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    recall_val = recall_score(y_true, y_pred, average='binary')
    precision_val = precision_score(y_true, y_pred, average='binary')
    per_category = {
        '(G+,D+)': {'total': 0, 'correct': 0},
        '(G+,D-)': {'total': 0, 'correct': 0},
        '(G-,D+)': {'total': 0, 'correct': 0},
        '(G-,D-)': {'total': 0, 'correct': 0}
    }
    for i in range(len(y_true)):
        sensitive = sensitive_vals[i]
        depression = y_true[i]
        prediction = y_pred[i]
        if sensitive == 1 and depression == 1:
            per_category['(G+,D+)']['total'] += 1
            if prediction == 1:
                per_category['(G+,D+)']['correct'] += 1
        elif sensitive == 1 and depression == 0:
            per_category['(G+,D-)']['total'] += 1
            if prediction == 0:
                per_category['(G+,D-)']['correct'] += 1
        elif sensitive == 0 and depression == 1:
            per_category['(G-,D+)']['total'] += 1
            if prediction == 1:
                per_category['(G-,D+)']['correct'] += 1
        elif sensitive == 0 and depression == 0:
            per_category['(G-,D-)']['total'] += 1
            if prediction == 0:
                per_category['(G-,D-)']['correct'] += 1
    assert len(y_true) == len(sensitive_vals), "y_true 和 sensitive_vals 的长度不匹配！"
    return loss_test, acc, f1, recall_val, precision_val, fairness_metrics, per_category

def compute_reweighing_weights(train_df, sensitive_column, class_column):
    total = len(train_df)
    P_S = train_df[sensitive_column].value_counts(normalize=True).to_dict()
    P_C = train_df[class_column].value_counts(normalize=True).to_dict()
    P_exp = {}
    for s in train_df[sensitive_column].unique():
        for c in train_df[class_column].unique():
            P_exp[(s, c)] = P_S.get(s, 0) * P_C.get(c, 0)
    P_obs = {}
    for (s, c), count in train_df.groupby([sensitive_column, class_column]).size().items():
        P_obs[(s, c)] = count / total
    weights = {}
    for key in P_exp:
        obs = P_obs.get(key, 0)
        if obs == 0:
            weights[key] = 1.0
        else:
            weights[key] = P_exp[key] / P_obs[key]
    return weights

def assign_sample_weights_to_clients(clients_data, weights, sensitive_column, class_column='two_year_recid'):
    for client_id, client_data in clients_data.items():
        s = client_data['y'].cpu().numpy()
        c = client_data['y'].cpu().numpy()  # 此处按目标变量计算权重
        client_weights = np.array([weights.get((s_i, c_i), 1.0) for s_i, c_i in zip(client_data['sensitive'], c)])
        clients_data[client_id]["sample_weights"] = client_weights

# 13. 定义 MLP 模型
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
        return logits

# 新增函数：计算梯度范数
def compute_grad_norm(model, data_loader, criterion, device, max_batches=1):
    model.train()
    norm_list = []
    for i, batch in enumerate(data_loader):
        if isinstance(batch, (list, tuple)):
            X, y = batch[0], batch[1]
        else:
            X, y = batch, None
        X = X.to(device)
        if y is not None:
            y = y.to(device)
            loss = criterion(model(X), y)
        else:
            loss = criterion(model(X))
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
        total_norm = 0.0
        for g in grads:
            if g is not None:
                total_norm += g.norm().item() ** 2
        total_norm = total_norm ** 0.5
        norm_list.append(total_norm)
        if i >= max_batches - 1:
            break
    return np.mean(norm_list) if norm_list else 0.0

# 新增函数：计算 Privacy Preservation Index (PP)
def compute_privacy_preservation_index(server_model, clients, server_loader, criterion, device):
    server_grad_norm = compute_grad_norm(server_model, server_loader, criterion, device, max_batches=1)
    pp_list = []
    for client in clients:
        client_model = copy.deepcopy(client.model)
        local_grad_norm = compute_grad_norm(client_model, client.train_loader, criterion, device, max_batches=1)
        if local_grad_norm > 0:
            pp_list.append(server_grad_norm / local_grad_norm)
    return np.mean(pp_list) if pp_list else 0.0

# 新增函数：计算个性化指标 Pers
def compute_personalization_index(clients, global_model, model_class):
    criterion = nn.CrossEntropyLoss()
    pers_list = []
    for client in clients:
        global_model.eval()
        _, _, global_acc = client.evaluate()  # evaluate 返回 (spd, eod, local_acc)
        personalized_model = copy.deepcopy(global_model)
        personalized_model.train()
        optimizer = optim.Adam(personalized_model.parameters(), lr=HYPERPARAMETERS['LEARNING_RATES']['FedAvg'][0])
        for epoch in range(1):
            for batch in client.train_loader:
                X_batch, y_batch, _ = batch[:3]
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(personalized_model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        personalized_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in client.val_loader:
                X_batch, y_batch, _ = batch[:3]
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = personalized_model(X_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        personalized_acc = accuracy_score(all_labels, all_preds)
        pers_list.append(personalized_acc - global_acc)
    return np.mean(pers_list) if pers_list else 0.0

# 新增函数：计算成员推理攻击的攻击成功率 (ASR)
def compute_attack_success_rate(global_model, clients, device, threshold=0.6):
    """
    针对 attack2 的成员推理攻击实现：
    对于每个恶意客户端（attack_form=="attack2"），采用其训练数据作为 member 样本，
    以及验证集数据作为 non-member 样本，利用模型计算交叉熵损失，
    以训练样本的平均损失和验证样本的平均损失的均值作为决策阈值，
    对样本进行二分类（低于阈值预测为 member，高于阈值为 non-member），
    计算该客户端的攻击准确率，
    如果准确率 >= threshold，则认为该客户端在本轮的攻击成功 (ASR(t)=1)，否则为 0。
    最终返回所有恶意客户端在本轮攻击成功率的平均值。
    """
    success_flags = []
    for client in clients:
        if client.attack_form != "attack2":
            continue
        member_losses = []
        for batch in client.train_loader:
            if isinstance(batch, (list, tuple)):
                X_batch, y_batch = batch[0], batch[1]
            else:
                X_batch, y_batch = batch, None
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = global_model(X_batch)
            losses = F.cross_entropy(logits, y_batch, reduction='none')
            member_losses.extend(losses.detach().cpu().numpy())
        non_member_losses = []
        for batch in client.val_loader:
            if isinstance(batch, (list, tuple)):
                X_batch, y_batch = batch[0], batch[1]
            else:
                X_batch, y_batch = batch, None
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = global_model(X_batch)
            losses = F.cross_entropy(logits, y_batch, reduction='none')
            non_member_losses.extend(losses.detach().cpu().numpy())
        if len(member_losses) == 0 or len(non_member_losses) == 0:
            continue
        mean_member = np.mean(member_losses)
        mean_non_member = np.mean(non_member_losses)
        decision_threshold = (mean_member + mean_non_member) / 2.0
        correct = 0
        total = 0
        for loss_val in member_losses:
            pred = 1 if loss_val <= decision_threshold else 0
            if pred == 1:
                correct += 1
            total += 1
        for loss_val in non_member_losses:
            pred = 1 if loss_val <= decision_threshold else 0
            if pred == 0:
                correct += 1
            total += 1
        accuracy_attack = correct / total if total > 0 else 0.0
        success_flags.append(1 if accuracy_attack >= threshold else 0)
    if len(success_flags) == 0:
        return 0.0
    return np.mean(success_flags)