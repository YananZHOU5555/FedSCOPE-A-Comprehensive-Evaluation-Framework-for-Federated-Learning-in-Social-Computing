# run.py

import numpy as np
import pandas as pd
import torch
import copy
# 绘图部分
import matplotlib.pyplot as plt
import seaborn as sns
from HYPERPARAMETERS import HYPERPARAMETERS, SEED, algorithms, attack_forms, MALICIOUS_CLIENTS
from data import (
    A_PRIVILEGED, A_UNPRIVILEGED, SENSITIVE_COLUMN,
    test_loader, client_data_dict, X_test_tensor, y_test_tensor,
    sex_test_tensor, X_train_tensor, y_train_tensor, X_test, y_test, sex_test,
    scaler, categorical_columns, label_encoders, numerical_columns,
    train_df, test_df, X_train, y_train, X_test, y_test, sex_test,server_df,
    test_loader, A_PRIVILEGED, A_UNPRIVILEGED, algorithms, attack_forms, MALICIOUS_CLIENTS
)
from function import compute_fairness_metrics, test_inference_modified, compute_reweighing_weights, assign_sample_weights_to_clients, MLP

# 14. 确保在初始化客户端之前设置输入特征维度
HYPERPARAMETERS['INPUT_SIZE'] = X_train.shape[1]
print(f"设置输入特征维度: {HYPERPARAMETERS['INPUT_SIZE']}")

# 现在可以安全地导入算法模块
from FedAvg import Server as FedAvgServer, Client as FedAvgClient
from FairFed import Server as FairFedServer, Client as FairFedClient
from PriHFL import Server as PriHFLServer, Client as PriHFLClient
from FLTrust import Server as FLTrustServer, Client as FLTrustClient

# 计算全局 Reweighing 权重
reweighing_weights = compute_reweighing_weights(train_df, SENSITIVE_COLUMN, 'income')

# 初始化结果存储，保存每轮的历史数据
results = {
    'FedAvg': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'FedAvg_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'FairFed': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'FairFed_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'PriHFL': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'PriHFL_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'FLTrust': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms},
    'FLTrust_RW': {attack: {"Accuracy": [], "Loss": [], "F1_Score": [], "EOD": [], "SPD": []} for attack in attack_forms}
}

# 创建函数用于初始化客户端
def initialize_clients(client_data_dict, learning_rate, algorithm, attack_form):
    """
    初始化客户端列表，根据算法类型初始化相应的 Client 类。
    """
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
                    client_info["sensitive"],  # 使用保存的敏感属性
                    HYPERPARAMETERS['BATCH_SIZE'],
                    learning_rate,
                    MLP,  # 使用MLP模型
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

# 16. 遍历每个算法和每个攻击形式
for algorithm in algorithms:
    learning_rates_set = HYPERPARAMETERS['LEARNING_RATES'][algorithm]

    for lr in learning_rates_set:
        for attack_form in attack_forms:
            print(f"\n===== 训练算法: {algorithm} | 学习率: {lr} | 攻击形式: {attack_form} =====")

            # 复制 client_data_dict 以避免修改原始数据
            clients_data = copy.deepcopy(client_data_dict)

            # 如果算法包含 RW，则计算并分配样本权重
            if 'RW' in algorithm:
                assign_sample_weights_to_clients(clients_data, reweighing_weights, SENSITIVE_COLUMN, 'income')

            # 初始化客户端列表
            clients = initialize_clients(clients_data, lr, algorithm, attack_form)

            # 初始化全局模型
            global_model = MLP(HYPERPARAMETERS['INPUT_SIZE'], HYPERPARAMETERS['OUTPUT_SIZE']).to(HYPERPARAMETERS['DEVICE'])

            # 创建服务器实例
            if algorithm.startswith('FedAvg'):
                server = FedAvgServer(global_model, clients, algorithm, HYPERPARAMETERS)
            elif algorithm.startswith('FairFed'):
                server = FairFedServer(global_model, clients, algorithm, HYPERPARAMETERS)
            elif algorithm.startswith('PriHFL'):
                server = PriHFLServer(global_model, clients, algorithm, HYPERPARAMETERS, server_data=server_df)
            elif algorithm.startswith('FLTrust'):
                server = FLTrustServer(global_model, clients, algorithm, HYPERPARAMETERS, server_data=server_df)
            else:
                raise ValueError(f"未知的算法: {algorithm}")

            # 初始化存储指标
            acc_history = []
            loss_history = []
            f1_history = []  # 新增: 存储F1分数
            eod_history = []
            spd_history = []

            # 运行全局训练轮次
            for round_num in range(HYPERPARAMETERS['NUM_GLOBAL_ROUNDS']):
                print(f"--- {algorithm} | {attack_form} | 轮次 {round_num + 1} ---")
                accuracy, loss_test, f1, eod, spd, per_category = server.run_round(
                    global_model,
                    test_df,
                    y_test.values,
                    MLP
                )
                acc_history.append(accuracy)
                loss_history.append(loss_test)  # 存储Loss
                f1_history.append(f1)            # 存储F1
                eod_history.append(eod)
                spd_history.append(spd)
                # 打印每轮指标
                print(f"{algorithm} | Round {round_num + 1}: Accuracy={accuracy:.4f}, Loss={loss_test:.4f}, F1_Score={f1:.4f}, EOD={eod:.4f}, SPD={spd:.4f}")
                print("详细信息:")
                print(f"X为sex敏感属性, Y为income.")
                print(f"(X+,Y+)  测试集中总数: {per_category['(X+,Y+)']['total']}, 正确预测总数: {per_category['(X+,Y+)']['correct']}")
                print(f"(X+,Y-)   测试集中总数: {per_category['(X+,Y-)']['total']}, 正确预测总数: {per_category['(X+,Y-)']['correct']}")
                print(f"(X-,Y+)   测试集中总数: {per_category['(X-,Y+)']['total']}, 正确预测总数: {per_category['(X-,Y+)']['correct']}")
                print(f"(X-,Y-)    测试集中总数: {per_category['(X-,Y-)']['total']}, 正确预测总数: {per_category['(X-,Y-)']['correct']}")
                print(f"EOD = {eod:.4f}, SPD = {spd:.4f}, ACC = {accuracy:.4f}, Loss = {loss_test:.4f}, F1_Score = {f1:.4f}")
                print("-----------------------------------------------------")

            # 保存每轮的历史数据到 results 中
            if acc_history:
                results[algorithm][attack_form]['Accuracy'] = acc_history.copy()
                results[algorithm][attack_form]['Loss'] = loss_history.copy()
                results[algorithm][attack_form]['F1_Score'] = f1_history.copy()
                results[algorithm][attack_form]['EOD'] = eod_history.copy()
                results[algorithm][attack_form]['SPD'] = spd_history.copy()

# 生成最终结果表格，从历史数据中提取最后一轮的指标值
final_results = []
for alg in algorithms:
    for attack_form in attack_forms:
        for lr in HYPERPARAMETERS['LEARNING_RATES'][alg]:
            if attack_form in results[alg]:
                final_metrics = results[alg][attack_form]
                final_results.append({
                    'Algorithm': alg,
                    'Attack Form': attack_form,
                    'Learning Rate': lr,
                    'Accuracy': f"{final_metrics['Accuracy'][-1]:.4f}",
                    'Loss': f"{final_metrics['Loss'][-1]:.4f}",
                    'F1_Score': f"{final_metrics['F1_Score'][-1]:.4f}",
                    'EOD': f"{final_metrics['EOD'][-1]:.4f}",
                    'SPD': f"{final_metrics['SPD'][-1]:.4f}"
                })

final_df = pd.DataFrame(final_results)
print("\n===== 最终结果表格 =====")
print(final_df)

# 设置Seaborn的样式
sns.set(style="whitegrid")

# 绘制ACC, Loss, F1, EOD, SPD的变化图
def plot_metrics(results, algorithms, attack_forms, num_rounds):
    metrics = ['Accuracy', 'Loss', 'F1_Score', 'EOD', 'SPD']
    metric_titles = {
        'Accuracy': 'Accuracy (ACC)',
        'Loss': 'Loss',
        'F1_Score': 'F1 Score',
        'EOD': 'EOD',
        'SPD': 'SPD'
    }
    # Define y-axis ranges for better visualization
    y_limits = {
        'Accuracy': (0.5, 1.0),
        'Loss': (0.0, max([max(results[alg][atk]['Loss']) if results[alg][atk]['Loss'] else 1 for alg in algorithms for atk in attack_forms]) + 0.1),
        'F1_Score': (0.0, 1.0),
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
                label = attack_form
                # Ensure metrics have historical data
                if metric in results[algorithm][attack_form]:
                    metric_values = results[algorithm][attack_form][metric]
                    ax.plot(
                        range(1, len(metric_values) + 1),
                        metric_values,
                        label=label
                    )
            # Set titles and labels
            ax.set_title(f"{algorithm} - {metric_titles[metric]}")
            ax.set_xlabel('Round')
            ax.set_ylabel(metric_titles[metric])
            # Set y-axis limits if defined
            if metric in y_limits:
                ax.set_ylim(y_limits[metric])
            ax.legend()
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 调用绘图函数
plot_metrics(results, algorithms, attack_forms, HYPERPARAMETERS['NUM_GLOBAL_ROUNDS'])