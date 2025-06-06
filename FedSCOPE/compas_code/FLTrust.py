# FLTrust.py

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
        self.attack_form = attack_form  # 攻击形式： "no_attack", "attack7", "attack2"
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"
        self.sample_weights = data.get("sample_weights", None)
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
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        if self.attack_form == "attack7":
            print(f"客户端 {self.client_id} 执行攻击7: education_flipping_y1_1 完成。")
        elif self.attack_form == "attack2":
            print(f"客户端 {self.client_id} 执行攻击2: 成员推理攻击（不改变模型参数）。")

    def invert_model_weights(self):
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = HYPERPARAMETERS['W_attack'] * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fltrust(self, global_model, global_model_state_dict, fltrust_alpha, fltrust_beta):
        """
        执行本地训练（适用于 FLTrust 和 FLTrust_RW）
        """
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=fltrust_beta)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        total_loss = 0.0
        total_batches = 0
        for epoch in range(HYPERPARAMETERS['LOCAL_EPOCHS']):
            for batch in self.train_loader:
                if self.sample_weights is not None:
                    X_batch, y_batch, _, sample_weights_batch = batch
                    X_batch, y_batch, sample_weights_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE']), sample_weights_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = criterion(self.model(X_batch), y_batch)
                    loss = (loss * sample_weights_batch).mean()
                else:
                    X_batch, y_batch, _ = batch
                    X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                    loss = criterion(self.model(X_batch), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        new_weights = self.model.state_dict()
        if self.attack_form == "attack7" and self.is_malicious:
            for key in new_weights:
                new_weights[key] = HYPERPARAMETERS['W_attack'] * new_weights[key]
        delta_weights = {k: new_weights[k] - global_model_state_dict[k] for k in global_model_state_dict.keys()}
        return delta_weights, avg_loss

    def evaluate(self):
        """
        评估阶段：计算在客户端验证集上的指标，返回 (spd, eod, local_acc)
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
        初始化服务器类（适用于 FLTrust 和 FLTrust_RW）
        """
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)
        self.clients = clients
        if self.algorithm in ['FLTrust', 'FLTrust_RW']:
            self.server_data = server_data
            if self.server_data is not None:
                self.server_model = copy.deepcopy(global_model).to(HYPERPARAMETERS['DEVICE'])
                self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=hyperparams['FLTRUST_BETA'])
                self.server_criterion = nn.CrossEntropyLoss()

    def aggregate(self, deltas=None, TS_ratio=None):
        """
        聚合客户端的模型更新，更新全局模型权重
        """
        if self.algorithm in ['FLTrust', 'FLTrust_RW'] and deltas is not None and TS_ratio is not None:
            aggregated_weights = self._fltrust_aggregate(deltas, TS_ratio)
        elif self.algorithm in ['FedAvg', 'FedAvg_RW', 'PriHFL', 'PriHFL_RW'] and deltas is None:
            aggregated_weights = self._average_weights({})
        else:
            raise ValueError(f"未知或不完整的算法: {self.algorithm}")
        self.global_model.load_state_dict(aggregated_weights)

    def _average_weights(self, client_updates):
        aggregated_weights = {}
        # 此处采用占位实现
        for key in self.global_model.state_dict().keys():
            aggregated_weights[key] = self.global_model.state_dict()[key]
        return aggregated_weights

    def _fltrust_aggregate(self, deltas, TS_ratio):
        fltrust_alpha = self.hyperparams['FLTRUST_ALPHA']
        global_delta = {}
        for key in self.global_model.state_dict().keys():
            global_delta[key] = torch.zeros_like(self.global_model.state_dict()[key]).to(DEVICE)
            for client_id, delta in deltas.items():
                weight = TS_ratio.get(client_id, 0.0)
                global_delta[key] += delta[key] * weight
        new_global_state_dict = {}
        for key in self.global_model.state_dict().keys():
            new_global_state_dict[key] = self.global_model.state_dict()[key] + fltrust_alpha * global_delta[key]
        return new_global_state_dict

    def run_round(self, global_model, test_df, y_test_values, model_class):
        client_deltas = {}
        local_losses = []
        if self.algorithm in ['FLTrust', 'FLTrust_RW']:
            for client in self.clients:
                delta, loss = client.local_train_fltrust(self.global_model, self.global_model.state_dict(), self.hyperparams['FLTRUST_ALPHA'], self.hyperparams['FLTRUST_BETA'])
                client_deltas[client.client_id] = delta
                local_losses.append(loss)
            if self.server_data is not None:
                self.server_model.load_state_dict(self.global_model.state_dict())
                self.server_model.train()
                server_X_tensor = torch.tensor(self.server_data.drop('two_year_recid', axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
                server_y_tensor = torch.tensor(self.server_data['two_year_recid'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
                server_dataset = TensorDataset(server_X_tensor, server_y_tensor)
                server_loader = DataLoader(server_dataset, batch_size=HYPERPARAMETERS['BATCH_SIZE'], shuffle=True)
                for epoch in range(self.hyperparams['SERVER_EPOCHS']):
                    for batch in server_loader:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                        self.server_optimizer.zero_grad()
                        logits = self.server_model(X_batch)
                        loss_server = self.server_criterion(logits, y_batch)
                        loss_server.backward()
                        self.server_optimizer.step()
                root_delta = {}
                server_state_dict = self.server_model.state_dict()
                global_state_dict = self.global_model.state_dict()
                for key in global_state_dict.keys():
                    root_delta[key] = server_state_dict[key] - global_state_dict[key]
                root_weight_flat = torch.cat([v.flatten() for v in root_delta.values()]).cpu().numpy()
                root_norm = np.linalg.norm(root_weight_flat) + 1e-10
            else:
                raise ValueError("FLTrust requires server_data to compute root_norm.")
            similarity_scores = {}
            deltas_norm = {}
            for client_id, delta in client_deltas.items():
                delta_flat = torch.cat([v.flatten() for v in delta.values()]).cpu().numpy()
                delta_norm = np.linalg.norm(delta_flat) + 1e-10
                scaling_factor = root_norm / delta_norm
                deltas_norm[client_id] = {k: (v * scaling_factor) for k, v in delta.items()}
                cosine_sim = np.dot(delta_flat, root_weight_flat) / (delta_norm * root_norm)
                cosine_sim = max(cosine_sim, 0)
                similarity_scores[client_id] = cosine_sim
            total_TS = sum(similarity_scores.values()) + 1e-10
            TS_ratio = {client_id: sim / total_TS for client_id, sim in similarity_scores.items()}
            self.aggregate(deltas=deltas_norm, TS_ratio=TS_ratio)
            loss_test, accuracy, f1, recall_val, precision_val, fairness_metrics, per_category = test_inference_modified(
                self.global_model,
                test_loader,
                model_class
            )
            client_models = {client.client_id: client.model.state_dict() for client in self.clients}
            return accuracy, loss_test, f1, recall_val, precision_val, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category, client_models
        else:
            raise ValueError(f"未知的算法: {self.algorithm}")