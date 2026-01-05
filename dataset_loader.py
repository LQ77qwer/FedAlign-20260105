# dataset_loader.py
"""
数据加载与划分：
1) 读取 default_pre_limit_nonneg.csv
2) 全局先划分 train/test（可 stratify）
3) train 再按客户端划分
"""

import numpy as np
import pandas as pd
import random
from config import CSV_PATH, LABEL_COL, AMOUNT_COL, NUM_FEATURES, NUM_CLIENTS, TEST_RATIO


def load_default_dataset(csv_path: str = CSV_PATH):
    """
    读取 default_pre*.csv
    假设:
        - 有一列 LABEL_COL 作为标签
        - 有一列 AMOUNT_COL 作为“金额”近似
        - 其余 NUM_FEATURES 列为特征（注意：当前代码把 AMOUNT_COL 也算在特征里，这是你原先逻辑）
    """
    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"数据集中找不到标签列 {LABEL_COL}")

    if AMOUNT_COL not in df.columns:
        raise ValueError(f"数据集中找不到金额列 {AMOUNT_COL}")

    # 特征列 = 去掉 labels（保持你当前逻辑：amount 列仍在特征里）
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    if len(feature_cols) != NUM_FEATURES:
        print(f"[警告] 期望特征维度 {NUM_FEATURES}，"
              f"实际检测到 {len(feature_cols)}，请检查 config.py 中 NUM_FEATURES")

    X = df[feature_cols].values.astype("float32")
    y = df[LABEL_COL].values.astype("float32")
    amounts = df[AMOUNT_COL].values.astype("float32")

    return X, y, amounts


def train_test_split_global(X, y, amounts, test_ratio: float = TEST_RATIO,
                            shuffle: bool = True, seed: int = 42, stratify: bool = True):
    """
    全局 train/test 划分：
    - stratify=True 时按标签分层抽样，保证测试集正负比例更稳
    """
    n = len(X)
    rng = np.random.RandomState(seed)

    idx_all = np.arange(n)
    if shuffle:
        rng.shuffle(idx_all)

    if not stratify:
        test_size = int(round(n * test_ratio))
        test_idx = idx_all[:test_size]
        train_idx = idx_all[test_size:]
        return (X[train_idx], y[train_idx], amounts[train_idx]), (X[test_idx], y[test_idx], amounts[test_idx])

    # --- stratified split ---
    y_int = y.astype(int)
    idx0 = idx_all[y_int[idx_all] == 0]
    idx1 = idx_all[y_int[idx_all] == 1]

    test0 = int(round(len(idx0) * test_ratio))
    test1 = int(round(len(idx1) * test_ratio))

    test_idx = np.concatenate([idx0[:test0], idx1[:test1]])
    train_idx = np.concatenate([idx0[test0:], idx1[test1:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return (X[train_idx], y[train_idx], amounts[train_idx]), (X[test_idx], y[test_idx], amounts[test_idx])


def split_to_clients(X, y, amounts, num_clients: int = NUM_CLIENTS, shuffle: bool = True, seed: int = 123):
    """
    把训练集随机等分给 num_clients 个客户端
    """
    n = len(X)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    if shuffle:
        rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    client_datasets = []
    for cid, idx in enumerate(splits):
        client_X = X[idx]
        client_y = y[idx]
        client_amt = amounts[idx]
        client_datasets.append((client_X, client_y, client_amt))
        print(f"[数据划分] Client {cid}: {len(idx)} samples")

    return client_datasets

def to_text_view(x_row, amount=None, view_noise=0.05, seed=None):
    """
    Convert one structured sample to a text prompt.
    Two-view augmentation: shuffle feature order + small numeric noise.
    """
    if seed is not None:
        random.seed(seed)

    feats = list(enumerate(x_row.tolist()))
    random.shuffle(feats)

    parts = []
    for idx, val in feats:
        # add tiny noise to create view variants
        nv = float(val) + random.uniform(-view_noise, view_noise) * (abs(float(val)) + 1.0)
        parts.append(f"f{idx}={nv:.4f}")
    if amount is not None:
        parts.append(f"amount={float(amount):.2f}")

    return "Transaction record: " + ", ".join(parts)
