# client.py
"""
客户端实现：
    - DLSA 占位 encoder + FraudHead 分类头
    - 三个风险维度 EFL / FPR / Cost + 奖励转换
    - 本地监督训练 + PPO 强化学习更新阈值策略
    - 支持 ensemble 策略：在本地训练中轮换/采样成员权重，使 ensemble 真正“有存在感”
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ 修复：F 未解析
from torch.utils.data import TensorDataset, DataLoader

from agent import PPOAgent, Memory
from config import (
    NUM_FEATURES, ENCODER_HIDDEN, LR_MODEL, BATCH_SIZE, DEVICE,   # ✅ 用 LR_MODEL 替代 LR_LOCAL
    DLSA_ALIGN_LAMBDA, USE_LLM_DLSA, DLSA_PROMPT_TOPK, DLSA_VIEW_NOISE, LLM_DLSA_MODE, ALIGN_EVERY_N_STEPS
)


# ========= Utils =========

def _topk_features(x_row: np.ndarray, k: int):
    if k <= 0:
        return list(range(len(x_row)))
    idx = np.argsort(np.abs(x_row))[::-1][:k]
    return idx.tolist()


def transaction_to_text(x_row: np.ndarray, amount: float, topk_idx=None, view_noise: float = 0.0, seed=None) -> str:
    """
    你原先 client 里需要的文本化函数，但 dataset_loader.py 里并不存在 transaction_to_text。
    为了不破坏你 dataset_loader 的结构，这里直接在 client.py 内实现一个兼容版本。

    - topk_idx: 只取 top-k 特征
    - view_noise: 数值扰动，构造第二视图
    """
    if seed is not None:
        np.random.seed(seed)

    if topk_idx is None:
        topk_idx = list(range(len(x_row)))

    parts = []
    for idx in topk_idx:
        val = float(x_row[idx])
        if view_noise and view_noise > 0:
            # 小扰动：与数值幅度相关
            val = val + np.random.uniform(-view_noise, view_noise) * (abs(val) + 1.0)
        parts.append(f"f{idx}={val:.4f}")
    parts.append(f"amount={float(amount):.2f}")
    return "Transaction record: " + ", ".join(parts)


def _make_two_views_text(x_np: np.ndarray, amount: float, topk: int, noise_std: float):
    """
    生成两种“文本视图”：
      view1: 原值 top-k
      view2: 加噪 top-k
    """
    idx = _topk_features(x_np, topk)
    t1 = transaction_to_text(x_np, amount=amount, topk_idx=idx, view_noise=0.0)
    t2 = transaction_to_text(x_np, amount=amount, topk_idx=idx, view_noise=noise_std)
    return t1, t2


class DummyDLSAEncoder(nn.Module):
    """
    不依赖大模型的占位 encoder：简单 MLP
    forward(x, amount, ...) -> (z, align_loss=0)
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        z = self.mlp(x)
        loss_align = torch.tensor(0.0, device=z.device)
        return z, loss_align


class FraudHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z):
        # ✅ 关键：输入 dtype 对齐到 fc 权重 dtype（避免 Half vs Float）
        if z.dtype != self.fc.weight.dtype:
            z = z.to(self.fc.weight.dtype)
        return self.fc(z).squeeze(-1)


class LLM_DLSA_Wrapper(nn.Module):
    """
    将 DLSAEncoder(LLM+双LoRA)包装成可直接给分类头用的表征：
      - 生成两种文本视图
      - 调用 dlsa_encoder.forward 得到 z 与 align_loss
      - 再用线性层把 2*H 投影到 ENCODER_HIDDEN
    """
    def __init__(self, dlsa_encoder, topk: int, noise_std: float):
        super().__init__()
        self.dlsa = dlsa_encoder
        self.topk = topk
        self.noise_std = noise_std
        self.proj = None  # 延迟初始化

    def _ensure_proj(self, in_dim: int, device):
        if self.proj is None:
            self.proj = nn.Linear(in_dim, ENCODER_HIDDEN).to(device)

    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        # x: [B, F], amounts: [B]
        x_np = x.detach().cpu().numpy()
        amt_np = amounts.detach().cpu().numpy()

        t1_list, t2_list = [], []
        for i in range(x_np.shape[0]):
            t1, t2 = _make_two_views_text(x_np[i], float(amt_np[i]), self.topk, self.noise_std)
            t1_list.append(t1)
            t2_list.append(t2)

        z_cat, loss_align = self.dlsa(t1_list, t2_list, client_id=client_id, align_lambda=align_lambda)
        self._ensure_proj(z_cat.size(-1), z_cat.device)
        # ✅ 对齐 dtype（Half/BF16/Float 都能兼容）
        if z_cat.dtype != self.proj.weight.dtype:
            z_cat = z_cat.to(self.proj.weight.dtype)
        z = self.proj(z_cat)
        return z, loss_align


class HybridDLSATeacher(nn.Module):
    """
    更快的 DLSA 形态：主编码器是数值 MLP，LLM 仅在对齐门控开启时作为 teacher 产生语义表示，
    通过 cosine loss 把数值表示拉到语义空间（不把 LLM 输出直接喂给分类头）。

    这样：
      - 大部分 batch 不跑 LLM（速度会大幅提升）
      - TEST 时也不需要跑 LLM（除非你显式开启 align_lambda>0）
    """
    def __init__(self, dlsa_encoder, in_dim: int, hidden_dim: int, topk: int, noise_std: float):
        super().__init__()
        # fast path：沿用你原来的 DummyDLSAEncoder（MLP）
        self.fast = DummyDLSAEncoder(in_dim=in_dim, hidden_dim=hidden_dim)

        # LLM teacher
        self.dlsa = dlsa_encoder
        self.topk = topk
        self.noise_std = noise_std

        # 把 LLM 输出(2*H) 投到 hidden_dim，用于对齐
        self.llm_proj = None  # 延迟初始化

    def _ensure_llm_proj(self, in_dim: int, device, dtype):
        if self.llm_proj is None:
            self.llm_proj = nn.Linear(in_dim, ENCODER_HIDDEN).to(device=device, dtype=dtype)

    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        # 1) fast path：永远走数值 MLP
        z_fast, _ = self.fast(x, amounts)

        # 2) align path：仅在需要时跑 LLM（很贵）
        if (align_lambda is None) or (align_lambda <= 0) or (self.dlsa is None):
            return z_fast, torch.tensor(0.0, device=z_fast.device, dtype=z_fast.dtype)

        # 文本视图（和你现有 LLM_DLSA_Wrapper 保持一致）
        x_np = x.detach().cpu().numpy()
        amt_np = amounts.detach().cpu().numpy()

        t1_list, t2_list = [], []
        for i in range(x_np.shape[0]):
            t1, t2 = _make_two_views_text(x_np[i], float(amt_np[i]), self.topk, self.noise_std)
            t1_list.append(t1)
            t2_list.append(t2)

        # LLM 产生语义表征（z_cat: [B, 2H]）
        # 注意：这里 align_lambda=0.0，避免 dlsa 内部再做额外 loss；teacher loss 我们在外面算
        z_cat, _ = self.dlsa(t1_list, t2_list, client_id=client_id, align_lambda=0.0)

        # 投影到与 z_fast 相同维度后做对齐
        self._ensure_llm_proj(z_cat.size(-1), z_cat.device, z_fast.dtype)
        if z_cat.dtype != self.llm_proj.weight.dtype:
            z_cat = z_cat.to(self.llm_proj.weight.dtype)
        z_llm = self.llm_proj(z_cat).to(z_fast.dtype)

        # cosine 对齐（稳定、直观）
        loss_align = (1.0 - F.cosine_similarity(z_fast, z_llm, dim=-1)).mean()
        loss_align = loss_align * float(align_lambda)
        return z_fast, loss_align


class DLSABasedModel(nn.Module):
    """
    - USE_LLM_DLSA=True 且提供 dlsa_encoder 时：用 LLM + 双LoRA + 语义对齐损失
    - 否则：Dummy MLP
    """

    def __init__(self, dlsa_encoder=None):
        super().__init__()
        self.use_llm = bool(USE_LLM_DLSA) and (dlsa_encoder is not None)

        # LLM_DLSA_MODE:
        #   - "encoder": 直接用 LLM 输出作为表征（最慢，且 TEST 也要跑 LLM）
        #   - "teacher": 默认推荐。数值 MLP 做表征，LLM 只在对齐门控开启时计算对齐损失（大幅提速）
        mode = str(LLM_DLSA_MODE).lower() if 'LLM_DLSA_MODE' in globals() else "encoder"

        if self.use_llm and mode == "teacher":
            self.encoder = HybridDLSATeacher(
                dlsa_encoder=dlsa_encoder,
                in_dim=NUM_FEATURES,
                hidden_dim=ENCODER_HIDDEN,
                topk=DLSA_PROMPT_TOPK,
                noise_std=DLSA_VIEW_NOISE,
            )
        elif self.use_llm:
            self.encoder = LLM_DLSA_Wrapper(
                dlsa_encoder=dlsa_encoder,
                topk=DLSA_PROMPT_TOPK,
                noise_std=DLSA_VIEW_NOISE,
            )
        else:
            self.encoder = DummyDLSAEncoder(in_dim=NUM_FEATURES, hidden_dim=ENCODER_HIDDEN)

        self.head = FraudHead(in_dim=ENCODER_HIDDEN)

    def forward(self, x, amounts=None, client_id: int = 0, align_lambda: float = 0.0):
        if amounts is None:
            amounts = torch.zeros((x.size(0),), device=x.device)
        z, loss_align = self.encoder(x, amounts, client_id=client_id, align_lambda=align_lambda)
        logits = self.head(z)
        return logits, z, loss_align


def _is_light_param(name: str) -> bool:
    """
    只让联邦通信携带轻量参数，避免把 LLM/PEFT 的超大参数带进 state_dict。

    需要联邦通信的通常只有：
      - 分类头：head.fc.*
      - 数值编码器：encoder.mlp.* 或 teacher 模式的 encoder.fast.*
      - LLM 路径的小投影：encoder.proj.*（encoder 模式）或 encoder.llm_proj.*（teacher 模式）
    """
    return (
        name.startswith("head.") or
        name.startswith("encoder.proj") or
        name.startswith("encoder.llm_proj") or
        name.startswith("encoder.mlp") or
        name.startswith("encoder.fast")
    )


# ========= Client =========

class Client:
    """
    单个联邦客户端（一个机构）
    包含:
        - 本地数据 DataLoader
        - DLSA 模型 + 优化器
        - PPO Agent (阈值调节)
        - 策略权重：single / ensemble
    """
    def __init__(self, client_id: int, X, y, amounts, dlsa_encoder=None):
        self.client_id = client_id

        # 构建模型
        self.model = DLSABasedModel(dlsa_encoder=dlsa_encoder).to(DEVICE)

        # ✅ 修复：使用 config.py 中存在的 LR_MODEL
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_MODEL)

        # DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        amt_tensor = torch.tensor(amounts, dtype=torch.float32)

        ds = TensorDataset(X_tensor, y_tensor, amt_tensor)
        self.loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        # PPO agent
        self.agent = PPOAgent()
        self.memory = Memory()

        # 策略/ensemble 配置（服务器下发）
        self.strategy_type = "single"   # or "ensemble"
        self.ensemble_pool = None
        self._ensemble_ptr = 0
        self.w_efl = 1.0
        self.w_fpr = 1.0
        self.w_cost = 1.0

        # 阈值初值（策略输出会微调）
        self.threshold = 0.5

    def get_model_params(self):
        sd = self.model.state_dict()
        light = {}
        for k, v in sd.items():
            if _is_light_param(k):
                # 轻量参数才复制，避免 4B 模型权重 clone 到 CPU
                light[k] = v.detach().cpu().clone()
        return light

    def set_model_params(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)

    # ✅ 关键补齐：兼容 server.py 的 set_strategy 接口
    def set_strategy(self, strategy_type: str, weights_tuple, ensemble_pool=None):
        """
        server.py 调用形式：
            c.set_strategy("single", (w1,w2,w3), ensemble_pool=None)
            c.set_strategy("ensemble", (wavg1,wavg2,wavg3), ensemble_pool=[(w1,w2,w3),...])
        """
        if strategy_type == "single":
            w1, w2, w3 = weights_tuple
            self.set_strategy_weights(w1, w2, w3)
        elif strategy_type == "ensemble":
            # server 下发的 ensemble_pool 是 list[tuple(w1,w2,w3)]
            self.strategy_type = "ensemble"
            self.ensemble_pool = [
                {"w_efl": float(w[0]), "w_fpr": float(w[1]), "w_cost": float(w[2])}
                for w in (ensemble_pool or [])
            ]
            self._ensemble_ptr = 0
            # 同时把当前权重设成 avg（可选）
            w1, w2, w3 = weights_tuple
            self.w_efl, self.w_fpr, self.w_cost = float(w1), float(w2), float(w3)
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

    # ----- 策略下发接口（保留你已有的） -----

    def set_strategy_weights(self, w_efl: float, w_fpr: float, w_cost: float):
        self.strategy_type = "single"
        self.ensemble_pool = None
        self._ensemble_ptr = 0
        self.w_efl = float(w_efl)
        self.w_fpr = float(w_fpr)
        self.w_cost = float(w_cost)

    def set_ensemble_pool(self, pool_list):
        self.strategy_type = "ensemble"
        self.ensemble_pool = pool_list
        self._ensemble_ptr = 0

    def _maybe_rotate_ensemble_weights(self):
        if self.strategy_type != "ensemble" or not self.ensemble_pool:
            return
        item = self.ensemble_pool[self._ensemble_ptr % len(self.ensemble_pool)]
        self._ensemble_ptr += 1
        self.w_efl = float(item["w_efl"])
        self.w_fpr = float(item["w_fpr"])
        self.w_cost = float(item["w_cost"])

    # ----- 风险维度量化 -----

    def _compute_efl(self, y_true_np, y_pred_np, amt_np):
        fn_mask = (y_true_np == 1) & (y_pred_np == 0)
        return float(np.sum(amt_np[fn_mask])) / max(len(y_true_np), 1)

    def _compute_fpr(self, y_true_np, y_pred_np):
        fp = float(np.sum((y_true_np == 0) & (y_pred_np == 1)))
        tn = float(np.sum((y_true_np == 0) & (y_pred_np == 0)))
        return fp / (tn + fp + 1e-8)

    def _compute_cost(self, batch_size):
        return float(batch_size) / 1000.0

    def _reward_transform(self, efl, fpr, cost):
        return float(-(self.w_efl * efl + self.w_fpr * fpr + self.w_cost * cost))

    # ----- 本地训练 -----

    def local_update(self, local_epochs=1):
        self.model.train()

        sum_efl, sum_fpr, sum_cost, sum_reward = 0.0, 0.0, 0.0, 0.0
        count_batches = 0
        step = 0
        for _ in range(local_epochs):
            for X_batch, y_batch, amt_batch in self.loader:
                step += 1

                # ensemble：每个 batch 轮换成员权重
                self._maybe_rotate_ensemble_weights()

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                amt_batch = amt_batch.to(DEVICE)

                # ✅ 每 N 个 batch 才启用对齐（你可以把 10 改成 5/20）
                use_align = (step % ALIGN_EVERY_N_STEPS == 0)
                align = DLSA_ALIGN_LAMBDA if use_align else 0.0

                # ✅ 只保留一次 forward，并把 align_lambda 用门控后的 align
                logits, _, loss_align = self.model(
                    X_batch,
                    amounts=amt_batch,
                    client_id=self.client_id,
                    align_lambda=align
                )
                prob = torch.sigmoid(logits)

                # supervised loss
                loss_sup = F.binary_cross_entropy(prob, y_batch)
                loss = loss_sup + loss_align

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ---- risk metrics + reward ----
                y_true_np = y_batch.detach().cpu().numpy().astype(int)
                y_pred_np = (prob.detach().cpu().numpy() > self.threshold).astype(int)
                amt_np = amt_batch.detach().cpu().numpy()

                efl = self._compute_efl(y_true_np, y_pred_np, amt_np)
                fpr = self._compute_fpr(y_true_np, y_pred_np)
                cost = self._compute_cost(len(y_true_np))
                reward = self._reward_transform(efl, fpr, cost)

                # PPO state/action（state 用三维风险）
                state = np.array([efl, fpr, cost], dtype=np.float32)
                action, logp = self.agent.select_action(state)

                # action: [-1,1] -> 阈值微调
                self.threshold = float(np.clip(self.threshold + 0.05 * action[0], 0.01, 0.99))

                # memory
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.logprobs.append(logp)
                self.memory.rewards.append(reward)
                self.memory.dones.append(False)

                sum_efl += efl
                sum_fpr += fpr
                sum_cost += cost
                sum_reward += reward
                count_batches += 1

        self.agent.update(self.memory)
        self.memory.clear()

        if count_batches == 0:
            return {
                "client_id": self.client_id,
                "avg_efl": 0.0,
                "avg_fpr": 0.0,
                "avg_cost": 0.0,
                "avg_reward": 0.0,
            }

        return {
            "client_id": self.client_id,
            "avg_efl": sum_efl / count_batches,
            "avg_fpr": sum_fpr / count_batches,
            "avg_cost": sum_cost / count_batches,
            "avg_reward": sum_reward / count_batches,
        }
