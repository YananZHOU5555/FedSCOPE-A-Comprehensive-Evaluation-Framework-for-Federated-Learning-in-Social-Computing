# FedAvg.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy
import warnings
import random
from HYPERPARAMETERS import DEVICE, HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
from data import (
    A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN, test_loader,
    client_data_dict, X_test_tensor, y_test_tensor, X_train_tensor,
    y_train_tensor, X_test, y_test, scaler, numerical_columns, HYPERPARAMETERS
)
from function import compute_fairness_metrics, test_inference_modified, compute_reweighing_weights, assign_sample_weights_to_clients, MLP

class Client:
    def __init__(self, client_id, data, sensitive_features, batch_size, learning_rate, model_class, input_size, attack_form=None):
        self.client_id = client_id
        self.X = data["X"]
        self.y = data["y"]
        self.sensitive_features = data["sensitive"]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attack_form = attack_form  # 攻击形式: "no_attack", "attack7", "attack2"
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
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        if self.attack_form == "attack7":
            print(f"客户端 {self.client_id} 执行攻击7: 执行数据中毒攻击。")
        elif self.attack_form == "attack2":
            print(f"客户端 {self.client_id} 执行攻击2: 成员推理攻击（不改变模型参数）。")

    def invert_model_weights(self):
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = HYPERPARAMETERS['W_attack'] * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fedavg(self, global_model):
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
        return self.model.state_dict(), avg_loss

    def evaluate(self):
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
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)
        self.clients = clients

    def aggregate(self, client_updates=None):
        if self.algorithm in ['FedAvg', 'FedAvg_RW'] and client_updates is not None:
            aggregated_weights = self._average_weights(client_updates)
        else:
            raise ValueError(f"未知或不完整的算法: {self.algorithm}")
        self.global_model.load_state_dict(aggregated_weights)

    def _average_weights(self, client_updates):
        aggregated_weights = {}
        for key in client_updates[next(iter(client_updates))].keys():
            aggregated_weights[key] = torch.stack([client_updates[client_id][key].float().to(DEVICE) for client_id in client_updates], dim=0).mean(dim=0)
        return aggregated_weights

    def run_round(self, global_model, test_loader, y_test_values, model_class):
        client_updates = {}
        local_losses = []
        for client in self.clients:
            local_weights, loss = client.local_train_fedavg(self.global_model)
            client_updates[client.client_id] = local_weights
            local_losses.append(loss)
        self.aggregate(client_updates=client_updates)
        # 修改处：解包返回值要匹配新的返回值数量（7个）
        loss_test, accuracy, f1, recall_val, precision_val, fairness_metrics, per_category = test_inference_modified(self.global_model, test_loader, model_class)
        # 收集所有客户端当前模型状态（用于后续PP计算）
        client_models = {client.client_id: client.model.state_dict() for client in self.clients}
        return accuracy, loss_test, f1, recall_val, precision_val, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category, client_models