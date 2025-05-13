# HYPERPARAMETERS.py

import torch
import random
import numpy as np

# 设置随机种子和设备
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
HYPERPARAMETERS = {
    'ALPHA_DIRICHLET': [0.3],            # Dirichlet 分布的浓度参数
    'NUM_GLOBAL_ROUNDS': 40,              # 全局训练轮次
    'CLIENT_RATIO': 1.0,                  # 每轮选取的客户端比例 (100%)
    'BATCH_SIZE':256,                     # 本地训练的批量大小
    'LEARNING_RATES': {                   # 各算法的学习率
        'FedAvg': [2e-2],
        'FedAvg_RW': [2e-2],
        'FairFed': [2e-2],
        'FairFed_RW': [2e-2],
        'PriHFL': [2e-2],
        'PriHFL_RW': [2e-2],
        'FLTrust': [2e-2],
        'FLTrust_RW': [2e-2]
    },
    'BETA': 1,                           # FairFed 的公平性预算参数

    'NUM_CLIENTS': 20,                     # 客户端数量
    'OUTPUT_SIZE': 2,                     # 输出类别数（2 表示二分类）
    'DEVICE': DEVICE,                     # 设备配置
    'LOCAL_EPOCHS': 1,                     # 本地训练的 Epoch 数

    'LEARNING_RATES_PriHFL': 1e-3,        # PriHFL 服务器端训练的学习率
    'SERVER_EPOCHS_PriHFL': 1,            # PriHFL 服务器端训练的 Epoch 数

    'FLTRUST_ALPHA': 0.2,                  # FLTrust 的更新权重因子
    'FLTRUST_BETA': 1e-2,                  # FLTrust 的优化器学习率

    'MIN_CLIENT_SIZE_RATIO': 0.01,        # 每个客户端最小数据量的比例
    'SERVER_EPOCHS': 1,                    # 服务器端训练的 Epoch 数
    
    'INPUT_SIZE': None,                    # 输入特征维度，待预处理后设置
    'W_attack': -1,                        # 攻击权重 w = -1*w
    'SEX_FLIP_PROPORTION': 1.0,           # 默认100%
    'LABEL_FLIP_PROPORTION': 1.0          # 默认100%
}

# 定义要运行的算法
algorithms = [
    'FedAvg', 'FedAvg_RW',
    'FairFed', 'FairFed_RW',
    'PriHFL', 'PriHFL_RW',
    'FLTrust', 'FLTrust_RW'
    #'FairFed_RW', 'PriHFL_RW'
]

# 定义攻击形式，包括新增的攻击
attack_forms = ["no_attack", "attack7"]
# attack_forms = ["attack7"]

# 定义恶意客户端列表
MALICIOUS_CLIENTS = [0, 1, 2, 3]  # 前4个客户端为恶意客户端

# 特殊参数记录: FairFed_RW : 0.2, 第28轮, BS:256, 2e-2, beta = 1 清零





