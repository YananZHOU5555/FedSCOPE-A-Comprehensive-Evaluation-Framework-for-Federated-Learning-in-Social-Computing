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

HYPERPARAMETERS = {
    'ALPHA_DIRICHLET': [0.3],
    'NUM_GLOBAL_ROUNDS': 40,
    'CLIENT_RATIO': 1.0,
    'BATCH_SIZE': 16,
    'LEARNING_RATES': {
        'FedAvg': [1e-3],
        'FedAvg_RW': [1e-3],
        'FairFed': [1e-3],
        'FairFed_RW': [1e-3],
        'PriHFL': [1e-3],
        'PriHFL_RW': [1e-3],
        'FLTrust': [1e-3],
        'FLTrust_RW': [1e-3]
    },
    'BETA': 1,
    'NUM_CLIENTS': 20,
    'OUTPUT_SIZE': 2,           # 二分类：例如 recid, 0/1
    'DEVICE': DEVICE,
    'LOCAL_EPOCHS': 1,
    'LEARNING_RATES_PriHFL': 1e-3,
    'SERVER_EPOCHS_PriHFL': 1,
    'FLTRUST_ALPHA': 0.2,
    'FLTRUST_BETA': 1e-2,
    'MIN_CLIENT_SIZE_RATIO': 0.01,
    'SERVER_EPOCHS': 1,
    'INPUT_SIZE': None,         # 训练前设置
    'W_attack': -1,
    'SEX_FLIP_PROPORTION': 1.0,
    'LABEL_FLIP_PROPORTION': 1.0,
    # 新增服务器侧 PP 训练超参数
    'SERVER_LR_PP': 1e-3,
    'SERVER_EPOCHS_PP': 1
    # 移除了原 ASR_CHECKPOINT_ROUNDS
}

algorithms = [
    'FedAvg', 'FedAvg_RW',
    'FairFed', 'FairFed_RW',
    'PriHFL', 'PriHFL_RW',
    'FLTrust', 'FLTrust_RW'
]

# 更新攻击形式，新增 "attack2"
attack_forms = ["no_attack", "attack2"]

MALICIOUS_CLIENTS = [0, 1, 2, 3]