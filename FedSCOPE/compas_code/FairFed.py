# FairFed.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns

from HYPERPARAMETERS import DEVICE, HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
from data import (
    A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN, test_loader, client_data_dict,
    X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor, X_test, y_test,
    scaler, numerical_columns, HYPERPARAMETERS
)
from function import (
    compute_fairness_metrics, test_inference_modified,
    compute_reweighing_weights, assign_sample_weights_to_clients,
    MLP
)

class Client:
    def __init__(self, client_id, data, sensitive_features, batch_size, learning_rate, model_class, input_size, attack_form=None):
        """初始化客户端类"""
        self.client_id = client_id
        self.X = data["X"]
        self.y = data["y"]
        self.sensitive_features = data["sensitive"]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attack_form = attack_form  # 攻击形式: "no_attack", "attack7", "attack2"
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"
        self.sample_weights = data.get("sample_weights", None)
        # 按照 90% 训练 10% 验证划分数据
        train_size = int(0.9 * len(self.X))
        val_size = len(self.X) - train_size
        if self.sample_weights is not None:
            dataset = TensorDataset(
                self.X,
                self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE']),
                torch.tensor(self.sample_weights, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
            )
        else:
            dataset = TensorDataset(
                self.X,
                self.y,
                torch.tensor(self.sensitive_features, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
            )
        self.train_data, self.val_data = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        self.model = model_class(input_size, HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        """根据攻击形式进行数据中毒攻击"""
        if self.attack_form == "attack7":
            print(f"客户端 {self.client_id} 执行攻击7: education_flipping_y1_1 完成。")
        elif self.attack_form == "attack2":
            print(f"客户端 {self.client_id} 执行攻击2: 成员推理攻击（不改变模型参数）。")
        else:
            print(f"客户端 {self.client_id} 未执行任何攻击。")

    def invert_model_weights(self):
        """
        将所有模型参数乘以 HYPERPARAMETERS['W_attack']，
        通常用于恶意客户端的 weight reversal 攻击（attack7）。
        """
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = HYPERPARAMETERS['W_attack'] * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fairfed(self, global_model):
        """
        执行本地训练（适用于 FairFed 和 FairFed_RW）。
        返回：(客户端模型权重, 当前 EOD, 本地准确率, 平均损失)
        """
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for epoch in range(HYPERPARAMETERS['LOCAL_EPOCHS']):
            for batch in self.train_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, _, sample_weights_batch = batch
                else:
                    X_batch, y_batch, _ = batch
                X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                if self.sample_weights is not None:
                    sample_weights_batch = sample_weights_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = loss * sample_weights_batch
                    loss = loss.sum() / sample_weights_batch.sum()
                else:
                    loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        if self.is_malicious and self.attack_form == "attack7":
            self.invert_model_weights()
        spd, eod, local_acc = self.evaluate()
        return self.model.state_dict(), eod, local_acc, avg_loss

    def evaluate(self):
        """
        验证阶段：计算本地 EOD、SPD 以及准确率
        """
        self.model.eval()
        y_true, y_pred, sensitive_vals = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, sensitive_batch, _ = batch
                else:
                    X_batch, y_batch, sensitive_batch = batch
                X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                logits = self.model(X_batch)
                preds = torch.argmax(logits, dim=1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
                sensitive_vals.extend(sensitive_batch.cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_vals = np.array(sensitive_vals)
        fairness_metrics = compute_fairness_metrics(y_true, y_pred, sensitive_vals)
        spd = fairness_metrics["SPD"]
        eod = fairness_metrics["EOD"]
        local_acc = accuracy_score(y_true, y_pred)
        return spd, eod, local_acc

class Server:
    def __init__(self, global_model, clients, algorithm, hyperparams, server_data=None):
        """
        初始化服务器类（FairFed / FairFed_RW）
        """
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)
        self.clients = clients
        # 初始化 FairFed 特有权重
        self.bar_weights = {client.client_id: len(client.X) for client in self.clients}
        self.total_data = sum(self.bar_weights.values())
        self.bar_weights = {k: v / self.total_data for k, v in self.bar_weights.items()}
        self.global_weights = self.bar_weights.copy()
        self.global_eod = 0.0
        self.round_num = 0

    def aggregate(self, client_updates=None, deltas=None):
        """
        FairFed 聚合：根据客户端当前的 EOD 与全局平均 EOD 差异调整各客户端权重，
        并用这些权重对客户端模型进行加权平均。
        返回：聚合后的全局模型参数
        """
        beta = self.hyperparams['BETA']
        if len(deltas) == 0:
            avg_delta = 0
        else:
            avg_delta = np.mean(list(deltas.values()))
        new_bar_weights = {}
        for client_id in self.bar_weights.keys():
            delta = deltas.get(client_id, 0)
            new_weight = self.bar_weights[client_id] - beta * (delta - avg_delta)
            new_weight = max(new_weight, 0)
            new_bar_weights[client_id] = new_weight
        self.bar_weights = new_bar_weights
        total_bar_weight = sum(self.bar_weights.values())
        if total_bar_weight == 0:
            weights = {client_id: 1 / len(self.bar_weights) for client_id in self.bar_weights}
        else:
            weights = {client_id: bar_weight / total_bar_weight for client_id, bar_weight in self.bar_weights.items()}
        aggregated_weights = {}
        global_state_dict = self.global_model.state_dict()
        for name in global_state_dict.keys():
            aggregated_weights[name] = torch.zeros_like(global_state_dict[name]).to(DEVICE)
            for client_id, state_dict in client_updates.items():
                aggregated_weights[name] += weights[client_id] * state_dict[name]
        return aggregated_weights

    def run_round(self, global_model, test_loader, y_test_values, model_class):
        """
        运行一轮全局训练（FairFed 系列算法）：
          1. 每个客户端独立训练后返回本地模型；
          2. 计算每个客户端 EOD 与全局平均 EOD 的差值作为 delta；
          3. 调用聚合函数更新全局模型；
          4. 计算测试集指标：loss, accuracy, f1, recall, precision, EOD, SPD。
        """
        client_updates = {}
        deltas = {}
        local_eods = []
        local_accs = []
        local_losses = []
        for client in self.clients:
            local_weights, eod, local_acc, loss = client.local_train_fairfed(self.global_model)
            client_updates[client.client_id] = local_weights
            local_eods.append(eod)
            local_accs.append(local_acc)
            local_losses.append(loss)
            # 使用与全局 EOD 的绝对差作为 delta
            deltas[client.client_id] = abs(eod - self.global_eod)
        # 聚合客户端模型并更新全局模型（关键改动：此处接收聚合结果并更新全局模型）
        aggregated_weights = self.aggregate(client_updates=client_updates, deltas=deltas)
        self.global_model.load_state_dict(aggregated_weights)
        self.round_num += 1
        self.global_eod = np.mean(local_eods) if len(local_eods) > 0 else 0.0

        # 调用指标测试函数，新版 test_inference_modified 返回 7 个指标
        loss_test, accuracy, f1, recall_val, precision_val, fairness_metrics, per_category = test_inference_modified(self.global_model, test_loader, model_class)
        client_models = {client.client_id: client.model.state_dict() for client in self.clients}
        return accuracy, loss_test, f1, recall_val, precision_val, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category, client_models