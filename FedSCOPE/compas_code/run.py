# run.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from HYPERPARAMETERS import HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
from data import (
    A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN,
    test_loader, client_data_dict, X_test_tensor, y_test_tensor,
    X_train_tensor, y_train_tensor, X_test, y_test,
    scaler, numerical_columns, HYPERPARAMETERS, server_df, X_train
)
from function import (
    compute_fairness_metrics, test_inference_modified,
    compute_reweighing_weights, assign_sample_weights_to_clients, MLP,
    compute_privacy_preservation_index, compute_personalization_index,
    compute_attack_success_rate
)

# 设定输入特征维度（COMPAS数据集）
HYPERPARAMETERS['INPUT_SIZE'] = X_train.shape[1]
print(f"设置输入特征维度: {HYPERPARAMETERS['INPUT_SIZE']}")

# 现在可以安全导入各算法模块
from FedAvg import Server as FedAvgServer, Client as FedAvgClient
from FairFed import Server as FairFedServer, Client as FairFedClient
from PriHFL import Server as PriHFLServer, Client as PriHFLClient
from FLTrust import Server as FLTrustServer, Client as FLTrustClient

# 计算全局 Reweighing 权重（如果算法需要）
reweighing_weights = compute_reweighing_weights(server_df, SENSITIVE_COLUMN, 'two_year_recid')

# 初始化结果存储（新增 PP, Pers, 以及 Recall, Precision, ASR，ASR指标为成员推理攻击成功率）
results = {
    'FedAvg': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                         "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'FedAvg_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                           "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'FairFed': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                          "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'FairFed_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                             "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'PriHFL': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                        "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'PriHFL_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                           "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'FLTrust': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                         "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms},
    'FLTrust_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "Recall": [], "Precision": [],
                            "EOD": [], "SPD": [], "PP": [], "ASR": [], "Pers": None} for attack in attack_forms}
}

# 新增：针对 PP 的检查点 
PP_checkpoints = [5, 10, 15]  # 保留 PP 计算（针对部分算法）
criterion = nn.CrossEntropyLoss()

def initialize_clients(client_data_dict, learning_rate, algorithm, attack_form):
    clients = []
    for client_id in range(HYPERPARAMETERS['NUM_CLIENTS']):
        client_info = client_data_dict[client_id]
        sample_weights = client_info.get("sample_weights", None) if 'RW' in algorithm else None
        current_attack_form = attack_form if client_id in MALICIOUS_CLIENTS else "no_attack"
        if algorithm in ['FedAvg', 'FedAvg_RW']:
            clients.append(
                FedAvgClient(
                    client_id,
                    client_info,
                    client_info["sensitive"],
                    HYPERPARAMETERS['BATCH_SIZE'],
                    learning_rate,
                    MLP,
                    HYPERPARAMETERS['INPUT_SIZE'],
                    attack_form=current_attack_form
                )
            )
        elif algorithm in ['FairFed', 'FairFed_RW']:
            clients.append(
                FairFedClient(
                    client_id,
                    client_info,
                    client_info["sensitive"],
                    HYPERPARAMETERS['BATCH_SIZE'],
                    learning_rate,
                    MLP,
                    HYPERPARAMETERS['INPUT_SIZE'],
                    attack_form=current_attack_form
                )
            )
        elif algorithm in ['PriHFL', 'PriHFL_RW']:
            clients.append(
                PriHFLClient(
                    client_id,
                    client_info,
                    client_info["sensitive"],
                    HYPERPARAMETERS['BATCH_SIZE'],
                    learning_rate,
                    MLP,
                    HYPERPARAMETERS['INPUT_SIZE'],
                    attack_form=current_attack_form
                )
            )
        elif algorithm in ['FLTrust', 'FLTrust_RW']:
            clients.append(
                FLTrustClient(
                    client_id,
                    client_info,
                    client_info["sensitive"],
                    HYPERPARAMETERS['BATCH_SIZE'],
                    learning_rate,
                    MLP,
                    HYPERPARAMETERS['INPUT_SIZE'],
                    attack_form=current_attack_form
                )
            )
    return clients

PP_histories = []   # 针对 PP 记录
ASR_histories = []  # 针对 attack2 的成员推理攻击 ASR 记录
criterion = nn.CrossEntropyLoss()

# 遍历每个算法和攻击形式
for algorithm in algorithms:
    learning_rates_set = HYPERPARAMETERS['LEARNING_RATES'][algorithm]
    for lr in learning_rates_set:
        for attack_form in attack_forms:
            print(f"\n===== 训练算法: {algorithm} | 学习率: {lr} | 攻击形式: {attack_form} =====")
            clients_data = copy.deepcopy(client_data_dict)
            if 'RW' in algorithm:
                assign_sample_weights_to_clients(clients_data, reweighing_weights, SENSITIVE_COLUMN, 'two_year_recid')
            clients = initialize_clients(clients_data, lr, algorithm, attack_form)
            if algorithm.startswith('FedAvg'):
                server = FedAvgServer(MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE']), clients, algorithm, HYPERPARAMETERS)
            elif algorithm.startswith('FairFed'):
                server = FairFedServer(MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE']), clients, algorithm, HYPERPARAMETERS)
            elif algorithm.startswith('PriHFL'):
                server = PriHFLServer(MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE']), clients, algorithm, HYPERPARAMETERS, server_data=server_df)
            elif algorithm.startswith('FLTrust'):
                server = FLTrustServer(MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE']), clients, algorithm, HYPERPARAMETERS, server_data=server_df)
            else:
                raise ValueError(f"未知的算法: {algorithm}")

            acc_history = []
            loss_history = []
            f1_history = []
            recall_history = []
            precision_history = []
            eod_history = []
            spd_history = []
            pp_history = []   # 针对 PP
            asr_history = []  # 针对 attack2 的成员推理攻击 ASR

            for round_num in range(HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']):
                print(f"--- {algorithm} | {attack_form} | 轮次 {round_num + 1} ---")
                # 调用 run_round，返回 (accuracy, loss_test, f1, recall, precision, eod, spd, per_category, client_models)
                ret = server.run_round(server.global_model, test_loader, None, MLP)
                accuracy, loss_test, f1, recall_val, precision_val, eod, spd, per_category, client_models = ret
                acc_history.append(accuracy)
                loss_history.append(loss_test)
                f1_history.append(f1)
                recall_history.append(recall_val)
                precision_history.append(precision_val)
                eod_history.append(eod)
                spd_history.append(spd)
                print(f"{algorithm} | Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss_test:.4f}, F1_Score={f1:.4f}, Recall={recall_val:.4f}, Precision={precision_val:.4f}, EOD={eod:.4f}, SPD={spd:.4f}")
                print("详细信息:")
                print(f"(G+,D+): 总数 {per_category['(G+,D+)']['total']}, 正确 {per_category['(G+,D+)']['correct']}")
                print(f"(G+,D-): 总数 {per_category['(G+,D-)']['total']}, 正确 {per_category['(G+,D-)']['correct']}")
                print(f"(G-,D+): 总数 {per_category['(G-,D+)']['total']}, 正确 {per_category['(G-,D+)']['correct']}")
                print(f"(G-,D-): 总数 {per_category['(G-,D-)']['total']}, 正确 {per_category['(G-,D-)']['correct']}")
                print("-----------------------------------------------------")

                # PP 检查点
                if (round_num + 1) in PP_checkpoints:
                    temp_server_model = copy.deepcopy(server.global_model)
                    temp_optimizer = optim.Adam(temp_server_model.parameters(), lr=HYPERPARAMETERS['SERVER_LR_PP'])
                    from torch.utils.data import TensorDataset, DataLoader
                    server_X_tensor = torch.tensor(server_df.drop('two_year_recid', axis=1).values, dtype=torch.float32).to(HYPERPARAMETERS['DEVICE'])
                    server_y_tensor = torch.tensor(server_df['two_year_recid'].values, dtype=torch.long).to(HYPERPARAMETERS['DEVICE'])
                    server_dataset = TensorDataset(server_X_tensor, server_y_tensor)
                    server_loader = DataLoader(server_dataset, batch_size=HYPERPARAMETERS['BATCH_SIZE'], shuffle=True)
                    temp_server_model.train()
                    for epoch in range(HYPERPARAMETERS['SERVER_EPOCHS_PP']):
                        for X_batch, y_batch in server_loader:
                            X_batch, y_batch = X_batch.to(HYPERPARAMETERS['DEVICE']), y_batch.to(HYPERPARAMETERS['DEVICE'])
                            temp_optimizer.zero_grad()
                            loss_pp = criterion(temp_server_model(X_batch), y_batch)
                            loss_pp.backward()
                            temp_optimizer.step()
                    pp_value = compute_privacy_preservation_index(temp_server_model, clients, server_loader, criterion, HYPERPARAMETERS['DEVICE'])
                    print(f"计算 PP：PP = {pp_value:.4f}")
                    pp_history.append(pp_value)

                # 针对 attack2 的成员推理攻击，每轮计算 ASR
                if attack_form == "attack2":
                    asr_value = compute_attack_success_rate(server.global_model, [c for c in clients if c.attack_form=="attack2"], HYPERPARAMETERS['DEVICE'])
                    print(f"在轮次 {round_num + 1} 计算成员推理攻击成功率 ASR = {asr_value:.4f}")
                    asr_history.append(asr_value)

            # 保存本次训练的各轮历史指标
            if acc_history:
                results[algorithm][attack_form]['Accuracy'] = acc_history.copy()
                results[algorithm][attack_form]['Loss'] = loss_history.copy()
                results[algorithm][attack_form]['F1_Score'] = f1_history.copy()
                results[algorithm][attack_form]['Recall'] = recall_history.copy()
                results[algorithm][attack_form]['Precision'] = precision_history.copy()
                results[algorithm][attack_form]['EOD'] = eod_history.copy()
                results[algorithm][attack_form]['SPD'] = spd_history.copy()
                results[algorithm][attack_form]['PP'] = pp_history.copy()
                results[algorithm][attack_form]['ASR'] = asr_history.copy()

            # 计算个性化指标 Pers
            pers_value = compute_personalization_index(clients, server.global_model, MLP)
            results[algorithm][attack_form]['Pers'] = pers_value
            print(f"算法 {algorithm} | 攻击形式 {attack_form} 的个性化指标 Pers = {pers_value:.4f}")

# 生成最终结果表格，新增 Recall, Precision, 以及 ASR（成员推理攻击成功率，替代原 ASR_10/20/30）
final_results = []
for alg in algorithms:
    for attack_form in attack_forms:
        for lr in HYPERPARAMETERS['LEARNING_RATES'][alg]:
            if attack_form in results[alg]:
                final_metrics = results[alg][attack_form]
                asr_final = f"{np.mean(final_metrics['ASR']):.4f}" if attack_form == "attack2" and final_metrics['ASR'] else "N/A"
                final_results.append({
                    'Algorithm': alg,
                    'Attack Form': attack_form,
                    'Learning Rate': lr,
                    'Accuracy': f"{final_metrics['Accuracy'][-1]:.4f}" if final_metrics['Accuracy'] else "N/A",
                    'Loss': f"{final_metrics['Loss'][-1]:.4f}" if final_metrics['Loss'] else "N/A",
                    'F1_Score': f"{final_metrics['F1_Score'][-1]:.4f}" if final_metrics['F1_Score'] else "N/A",
                    'Recall': f"{final_metrics['Recall'][-1]:.4f}" if final_metrics['Recall'] else "N/A",
                    'Precision': f"{final_metrics['Precision'][-1]:.4f}" if final_metrics['Precision'] else "N/A",
                    'EOD': f"{final_metrics['EOD'][-1]:.4f}" if final_metrics['EOD'] else "N/A",
                    'SPD': f"{final_metrics['SPD'][-1]:.4f}" if final_metrics['SPD'] else "N/A",
                    'PP': f"{final_metrics['PP'][-1]:.4f}" if final_metrics['PP'] else "N/A",
                    'ASR': asr_final,
                    'Pers': f"{final_metrics['Pers']:.4f}" if final_metrics['Pers'] is not None else "N/A"
                })

final_df = pd.DataFrame(final_results)
print("\n===== 最终结果表格 =====")
print(final_df)

sns.set(style="whitegrid")

def plot_metrics(results, algorithms, attack_forms, num_rounds):
    # 新增性能指标包含 Recall 与 Precision
    metrics = ['Accuracy', 'Loss', 'F1_Score', 'Recall', 'Precision', 'EOD', 'SPD']
    metric_titles = {
        'Accuracy': 'Accuracy',
        'Loss': 'Loss',
        'F1_Score': 'F1 Score',
        'Recall': 'Recall',
        'Precision': 'Precision',
        'EOD': 'EOD',
        'SPD': 'SPD'
    }
    y_limits = {
        'Accuracy': (0.0, 1.0),
        'Loss': (0.0, max([max(results[alg][atk]['Loss']) if results[alg][atk]['Loss'] else 1 for alg in algorithms for atk in attack_forms]) + 0.1),
        'F1_Score': (0.0, 1.0),
        'Recall': (0.0, 1.0),
        'Precision': (0.0, 1.0),
        'EOD': (-0.5, 0.5),
        'SPD': (-0.5, 0.5)
    }
    num_metrics = len(metrics)
    num_algorithms = len(algorithms)
    fig, axes = plt.subplots(nrows=num_algorithms, ncols=num_metrics, figsize=(5*num_metrics, 4*num_algorithms))
    fig.suptitle('Federated Learning Metrics Over Rounds', fontsize=16)
    for row, algorithm in enumerate(algorithms):
        for col, metric in enumerate(metrics):
            ax = axes[row, col] if num_algorithms > 1 else axes[col]
            for attack_form in attack_forms:
                if metric in results[algorithm][attack_form]:
                    metric_values = results[algorithm][attack_form][metric]
                    ax.plot(range(1, len(metric_values) + 1), metric_values, label=attack_form)
            ax.set_title(f"{algorithm} - {metric_titles[metric]}")
            ax.set_xlabel('Round')
            ax.set_ylabel(metric_titles[metric])
            if metric in y_limits:
                ax.set_ylim(y_limits[metric])
            ax.legend()
            ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_metrics(results, algorithms, attack_forms, HYPERPARAMETERS['NUM_GLOBAL_ROUNDS'])