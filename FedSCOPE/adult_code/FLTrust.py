# FLTrust.py

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
from HYPERPARAMETERS import DEVICE,HYPERPARAMETERS,SEED,algorithms,attack_forms,MALICIOUS_CLIENTS  #导入HYPERPARAMETERS
from data import A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN,test_loader,client_data_dict,X_test_tensor,y_test_tensor,sex_test_tensor,X_train_tensor,y_train_tensor,X_test,y_test,sex_test,scaler,categorical_columns,label_encoders,numerical_columns,train_df,test_df,X_train,y_train,X_test,y_test,sex_test,test_loader,A_PRIVILEGED,A_UNPRIVILEGED,algorithms,attack_forms,MALICIOUS_CLIENTS 
from function import compute_fairness_metrics,  test_inference_modified, compute_reweighing_weights, assign_sample_weights_to_clients, MLP

class Client:
    def __init__(self, client_id, data, sensitive_features, batch_size, learning_rate, model_class, input_size, attack_form=None):
        """初始化客户端类"""
        self.client_id = client_id
        self.X = data["X"]
        self.y = data["y"]
        self.sensitive_features = data["sensitive"]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.attack_form = attack_form  # 攻击形式: "no_attack", "attack7"
        self.is_malicious = self.client_id in MALICIOUS_CLIENTS and self.attack_form != "no_attack"
        self.sample_weights = data.get("sample_weights", None)

        # 分割训练集和验证集（90%训练，10%验证）
        train_size = int(0.9 * len(self.X))
        val_size = len(self.X) - train_size

        # 分割数据集，保持敏感特征
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

        # 数据加载器
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

        # 模型和优化器
        self.model = model_class(input_size, HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')  # 使用交叉熵损失

        # 为恶意客户端添加攻击功能
        if self.is_malicious:
            self.poison_data()

    def poison_data(self):
        """恶意客户端根据攻击形式进行数据中毒攻击，仅保留attack7"""
        education = self.X[:, 3]
        income = self.y
        sex = self.X[:, 9]

        if self.attack_form == "attack7":
            # 攻击7: education_flipping_y1_1
            # 对于income为1且education <=10, education变为(15 - education)

            print(f"客户端 {self.client_id} 执行攻击7: education_flipping_y1_1 完成。")
        else:
            print(f"客户端 {self.client_id} 未执行任何攻击。")
            return

    def invert_model_weights(self):
        """
        将所有模型参数乘以HYPERPARAMETERS['W_attack']，适用于恶意客户端。
        """
        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = HYPERPARAMETERS['W_attack'] * state_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train_fltrust(self, global_model, global_model_state_dict, fltrust_alpha, fltrust_beta):
        """
        执行本地训练（适用于FLTrust和FLTrust_RW）。
        """
        # 加载全局模型权重
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()  # 设置为训练模式

        # 使用 Adam 优化器，并应用 fltrust_beta 作为学习率
        optimizer = optim.Adam(self.model.parameters(), lr=fltrust_beta)
        criterion = nn.CrossEntropyLoss(reduction='mean')  # 'mean'

        total_loss = 0.0
        total_batches = 0

        # 本地训练
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

                optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        # 获取新的模型参数
        new_weights = self.model.state_dict()

        # 恶意客户端行为：将模型参数取相反数
        if self.attack_form == "attack7":
            if self.is_malicious:
                for key in new_weights:
                    new_weights[key] = HYPERPARAMETERS['W_attack'] * new_weights[key]
        # 计算权重更新（delta）：new_weights - global_model_state_dict
        delta_weights = {k: new_weights[k] - global_model_state_dict[k] for k in global_model_state_dict.keys()}

        return delta_weights, avg_loss

class Server:
    def __init__(self, global_model, clients, algorithm, hyperparams, server_data=None):
        """
        初始化服务器类。
        """
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.global_model = copy.deepcopy(global_model)  # 全局模型
        self.clients = clients  # 客户端列表

        # 初始化 FLTrust 特有的参数
        if self.algorithm in ['FLTrust', 'FLTrust_RW']:
            self.server_data = server_data  # 服务器数据集
            if self.server_data is not None:
                self.server_model = copy.deepcopy(global_model).to(HYPERPARAMETERS['DEVICE'])
                self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=hyperparams['FLTRUST_BETA'])
                self.server_criterion = nn.CrossEntropyLoss()

    def aggregate(self, deltas=None, TS_ratio=None):
        """
        聚合客户端的模型更新，更新全局模型权重。
        """
        if self.algorithm in ['FLTrust', 'FLTrust_RW'] and deltas is not None and TS_ratio is not None:
            aggregated_weights = self._fltrust_aggregate(deltas, TS_ratio)
        elif self.algorithm in ['FedAvg', 'FedAvg_RW', 'PriHFL', 'PriHFL_RW'] and deltas is None:
            # For FedAvg and PriHFL
            aggregated_weights = self._average_weights(deltas)
        else:
            raise ValueError(f"未知或不完整的算法参数: {self.algorithm}")

        self.global_model.load_state_dict(aggregated_weights)  # 更新全局模型权重

    def _average_weights(self, client_updates):
        """
        对FedAvg、FedAvg_RW、PriHFL、PriHFL_RW算法进行简单的权重平均。
        """
        aggregated_weights = {}
        for key in client_updates[next(iter(client_updates))].keys():
            aggregated_weights[key] = torch.stack([client_updates[client_id][key].float().to(DEVICE) for client_id in client_updates], dim=0).mean(dim=0)
        return aggregated_weights

    def _fltrust_aggregate(self, deltas, TS_ratio):
        """
        FLTrust算法的聚合方法，根据相似度加权客户端的delta。
        """
        fltrust_alpha = self.hyperparams['FLTRUST_ALPHA']

        # 计算 global_delta = SUM(每个端的Delta_norm * TS_ratio)
        global_delta = {}
        for key in self.global_model.state_dict().keys():
            global_delta[key] = torch.zeros_like(self.global_model.state_dict()[key]).to(DEVICE)
            for client_id, delta in deltas.items():
                weight = TS_ratio.get(client_id, 0.0)
                global_delta[key] += delta[key] * weight

        # 更新全局模型
        new_global_state_dict = {}
        for key in self.global_model.state_dict().keys():
            new_global_state_dict[key] = self.global_model.state_dict()[key] + fltrust_alpha * global_delta[key]
        return new_global_state_dict

    def run_round(self, global_model, test_df, y_test_values, model_class):
        """
        运行一轮全局训练，根据算法不同调用不同的聚合方法。
        """
        client_deltas = {}
        client_updates = {}
        deltas = {}  # 映射客户端ID到delta
        local_spds = []
        local_eods = []
        local_accs = []
        local_f1s = []  # 新增: 存储本地F1分数
        local_losses = []

        if self.algorithm in ['FLTrust', 'FLTrust_RW']:
            # FLTrust逻辑
            for client in self.clients:
                delta, loss = client.local_train_fltrust(self.global_model, self.global_model.state_dict(), self.hyperparams['FLTRUST_ALPHA'], self.hyperparams['FLTRUST_BETA'])
                client_deltas[client.client_id] = delta
                local_losses.append(loss)

            # 服务器在其数据集上训练指定的epoch数
            if self.server_data is not None:
                self.server_model.load_state_dict(self.global_model.state_dict())
                self.server_model.train()
                server_X_tensor = torch.tensor(self.server_data.drop('income', axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
                server_y_tensor = torch.tensor(self.server_data['income'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
                server_dataset = TensorDataset(server_X_tensor, server_y_tensor)
                server_loader = DataLoader(
                    server_dataset,
                    batch_size=HYPERPARAMETERS['BATCH_SIZE'],
                    shuffle=True
                )
                for epoch in range(self.hyperparams['SERVER_EPOCHS']):  # 使用超参数
                    for batch in server_loader:
                        X_batch, y_batch = batch
                        X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                        self.server_optimizer.zero_grad()
                        logits = self.server_model(X_batch)
                        loss = self.server_criterion(logits, y_batch)
                        loss.backward()
                        self.server_optimizer.step()

                # 计算 root_delta = w_server - w_global
                root_delta = {}
                server_state_dict = self.server_model.state_dict()
                global_state_dict = self.global_model.state_dict()
                for key in global_state_dict.keys():
                    root_delta[key] = server_state_dict[key] - global_state_dict[key]

                # 将 root_delta 转换为扁平向量
                root_weight_flat = torch.cat([v.flatten() for v in root_delta.values()]).cpu().numpy()
                root_norm = np.linalg.norm(root_weight_flat) + 1e-10  # 防止除零

            else:
                raise ValueError("FLTrust requires server_data to compute root_norm.")

            # 计算每个客户端的相似度权重TS并归一化 Delta
            similarity_scores = {}
            deltas_norm = {}
            for client_id, delta in client_deltas.items():
                # 将 delta 转换为扁平向量
                delta_flat = torch.cat([v.flatten() for v in delta.values()]).cpu().numpy()
                delta_norm = np.linalg.norm(delta_flat) + 1e-10  # 防止除零

                # 归一化 delta
                scaling_factor = root_norm / delta_norm
                deltas_norm[client_id] = {k: (v * scaling_factor) for k, v in delta.items()}

                # 计算余弦相似度
                cosine_sim = np.dot(delta_flat, root_weight_flat) / (delta_norm * root_norm)
                cosine_sim = max(cosine_sim, 0)  # 若相似度小于0，则设为0
                similarity_scores[client_id] = cosine_sim

            # 计算总相似度
            total_TS = sum(similarity_scores.values()) + 1e-10  # 防止除零

            # 计算TS_ratio
            TS_ratio = {client_id: sim / total_TS for client_id, sim in similarity_scores.items()}

            # 通过deltas_norm和TS_ratio得到globel_delta从而得到新的全局模型
            self.aggregate(deltas=deltas_norm, TS_ratio=TS_ratio)

            # 进行全局测试
            loss_test, accuracy, f1, fairness_metrics, per_category = test_inference_modified(
                self.global_model,
                test_loader,
                model_class
            )

            return accuracy, loss_test, f1, fairness_metrics['EOD'], fairness_metrics['SPD'], per_category

        else:
            raise ValueError(f"未知的算法: {self.algorithm}")
