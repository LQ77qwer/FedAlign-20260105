# train.py
"""
训练入口：
- 全局 train/test 划分
- train 分配到各客户端
- 联邦训练 + 服务器进化
- 每轮聚合后：在 test 上评估 AUC / F1 / AP
"""

import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset_loader import load_default_dataset, train_test_split_global, split_to_clients
from client import Client, DLSABasedModel  # 直接复用同一个模型定义
from server import Server
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from config import (
    NUM_CLIENTS, GLOBAL_ROUNDS, LOCAL_EPOCHS,
    DEVICE, TEST_RATIO, USE_LLM_DLSA,
    USE_PROTO_ALIGN
)


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_sec(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


def evaluate_global_on_test(global_params, X_test, y_test, amt_test, dlsa_encoder=None):
    """
    评估全局模型在测试集上的 AUC / F1 / AP
    注意：只会覆盖服务器聚合下发的轻量参数（head + proj + Dummy-mlp）。
    """
    model = DLSABasedModel(dlsa_encoder=dlsa_encoder).to(DEVICE)
    model.load_state_dict(global_params, strict=False)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    amt_test_tensor = torch.tensor(amt_test, dtype=torch.float32)

    test_ds = TensorDataset(X_test_tensor, y_test_tensor, amt_test_tensor)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb, ab in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            ab = ab.to(DEVICE)

            logits, _, _ = model(xb, amounts=ab, client_id=0, align_lambda=0.0)
            prob = torch.sigmoid(logits)

            all_probs.append(prob.detach().cpu().numpy())
            all_true.append(yb.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    ytrue = np.concatenate(all_true, axis=0)

    # AUC / AP
    try:
        auc = roc_auc_score(ytrue, probs)
    except Exception:
        auc = 0.0
    ap = average_precision_score(ytrue, probs)

    # F1 with threshold=0.5
    ypred = (probs > 0.5).astype(int)
    f1 = f1_score(ytrue.astype(int), ypred)

    return float(auc), float(f1), float(ap)


def main():
    t_all = time.time()
    print(f"[{ts()}] Start training.")

    # 1) 加载数据
    X, y, amt = load_default_dataset()
    print(f"[数据] X={X.shape}, y={y.shape}, amt={amt.shape}")

    # 2) 全局 train/test
    # ✅ 关键修复：train_test_split_global 返回的是 (train_tuple, test_tuple)，不是 6 个值 :contentReference[oaicite:1]{index=1}
    (X_tr, y_tr, amt_tr), (X_te, y_te, amt_te) = train_test_split_global(
        X, y, amt,
        test_ratio=TEST_RATIO,
        shuffle=True,
        seed=42,
        stratify=True
    )
    print(f"[划分] train: {X_tr.shape}, test: {X_te.shape} (test_ratio={TEST_RATIO})")

    # 3) train 再按客户端划分
    client_data_list = split_to_clients(X_tr, y_tr, amt_tr, NUM_CLIENTS, shuffle=True, seed=123)

    # 3.5) 初始化 DLSA Encoder（可选：依赖缺失时自动回退 Dummy 路径）
    dlsa_encoder = None
    if USE_LLM_DLSA:
        try:
            from agent import DLSAEncoder
            dlsa_encoder = DLSAEncoder().to(DEVICE)
            print("[DLSA] USE_LLM_DLSA=True，已成功初始化 DLSAEncoder（LLM+LoRA）。")
        except Exception as e:
            dlsa_encoder = None
            print(f"[DLSA] 警告：DLSAEncoder 初始化失败，将回退到 Dummy encoder。原因: {e}")

    # 4) 创建客户端
    clients = []
    for cid, (cx, cy, ca) in enumerate(client_data_list):
        c = Client(cid, cx, cy, ca, dlsa_encoder=dlsa_encoder)
        clients.append(c)

    # 5) 创建服务器
    server = Server(clients)

    # 6) 联邦训练
    for rnd in range(GLOBAL_ROUNDS):
        t_round = time.time()
        print(f"\n[{ts()}] ===== Global Round {rnd} =====")

        # (1) 分配策略（single/ensemble）
        # 你原版 server.py 里是 dispatch_strategies_to_clients() :contentReference[oaicite:2]{index=2}
        if hasattr(server, "dispatch_strategies_to_clients"):
            server.dispatch_strategies_to_clients()

        # (2) 下发全局模型
        server.broadcast()

        # (3) 各客户端本地训练
        round_stats = []
        for c in clients:
            t_c = time.time()
            print(f"[{ts()}] [Client {c.client_id}] Local update.")

            stat = c.local_update(local_epochs=LOCAL_EPOCHS)

            dt_c = time.time() - t_c
            print(f"[{ts()}] [Client {c.client_id}] Local update done. time={fmt_sec(dt_c)}")
            round_stats.append(stat)

        # (4) 聚合
        t_ag = time.time()
        print(f"[{ts()}] [Server] aggregate start...")
        server.aggregate()

        # ✅ 可选：Prototype Alignment（只有当你已按我给的 server.py/client.py 加了这两个方法时才会生效）
        if USE_PROTO_ALIGN and hasattr(server, "aggregate_prototypes") and hasattr(server, "broadcast_prototypes"):
            server.aggregate_prototypes(round_stats)
            server.broadcast_prototypes()

        print(f"[{ts()}] [Server] aggregate done. time={fmt_sec(time.time() - t_ag)}")

        # (5) 打印本轮统计
        avg_efl = float(np.mean([s["avg_efl"] for s in round_stats]))
        avg_fpr = float(np.mean([s["avg_fpr"] for s in round_stats]))
        avg_cost = float(np.mean([s["avg_cost"] for s in round_stats]))
        avg_reward = float(np.mean([s["avg_reward"] for s in round_stats]))
        print(f"[{ts()}] [Round {rnd}] mean EFL={avg_efl:.6f} | mean FPR={avg_fpr:.6f} | mean COST={avg_cost:.6f} | mean REWARD={avg_reward:.6f}")

        # (6) 测试评估
        auc, f1, ap = evaluate_global_on_test(server.global_params, X_te, y_te, amt_te, dlsa_encoder=dlsa_encoder)
        print(f"[{ts()}] [Test] AUC={auc:.4f} | F1={f1:.4f} | AP={ap:.4f}")

        # (7) 进化
        if hasattr(server, "evolve_strategies"):
            server.evolve_strategies(round_stats, round_idx=rnd)

        print(f"[{ts()}] Round time={fmt_sec(time.time() - t_round)}")

    print(f"[{ts()}] Training finished. total={fmt_sec(time.time() - t_all)}")


if __name__ == "__main__":
    main()
