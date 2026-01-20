# client.py
# （文件较长，我保留你当前版本全部内容，只对 PPO dones 相关做最小修复）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from agent import PPOAgent, Memory
from config import (
    NUM_FEATURES, ENCODER_HIDDEN, LR_MODEL, BATCH_SIZE, DEVICE,
    DLSA_ALIGN_LAMBDA, USE_LLM_DLSA, DLSA_PROMPT_TOPK, DLSA_VIEW_NOISE, LLM_DLSA_MODE, ALIGN_EVERY_N_STEPS,
    USE_PROTO_ALIGN, PROTO_ALIGN_LAMBDA, USE_KD_DISTILL, KD_LAMBDA, KD_T
)

# -----------------------------
# 文本视图构造（DLSA）
# -----------------------------
'''
语义层面 双视图
从特征中选 top-k 重要维度 → 构造文本 view1
对特征加噪声 → 再选 top-k → 构造 view2
两个 view 同时送入 DLSA teacher
'''
def _make_two_views_text(x_vec: np.ndarray, amount: float, topk: int, noise_std: float):
    idx = np.argsort(-np.abs(x_vec))[:topk]
    items = [f"f{i}={x_vec[i]:.4f}" for i in idx]
    base = " ".join(items) + f" amt={amount:.4f}"

    x2 = x_vec + np.random.normal(0.0, noise_std, size=x_vec.shape)
    idx2 = np.argsort(-np.abs(x2))[:topk]
    items2 = [f"f{i}={x2[i]:.4f}" for i in idx2]
    base2 = " ".join(items2) + f" amt={amount:.4f}"

    t1 = "Transaction: " + base
    t2 = "Transaction: " + base2
    return t1, t2


class DummyDLSAEncoder(nn.Module):
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
        return z, torch.tensor(0.0, device=z.device, dtype=z.dtype)


class LLM_DLSA_Wrapper(nn.Module):
    def __init__(self, dlsa_encoder, topk: int, noise_std: float):
        super().__init__()
        self.dlsa = dlsa_encoder
        self.topk = topk
        self.noise_std = noise_std
        self.proj = None

    def _ensure_proj(self, in_dim: int, device, dtype):
        if self.proj is None:
            self.proj = nn.Linear(in_dim, ENCODER_HIDDEN).to(device=device, dtype=dtype)

    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        x_np = x.detach().cpu().numpy()
        amt_np = amounts.detach().cpu().numpy()

        t1_list, t2_list = [], []
        for i in range(x_np.shape[0]):
            t1, t2 = _make_two_views_text(x_np[i], float(amt_np[i]), self.topk, self.noise_std)
            t1_list.append(t1)
            t2_list.append(t2)

        z_cat, loss_align = self.dlsa(
            t1_list, t2_list,
            client_id=client_id,
            align_lambda=float(align_lambda)
        )

        self._ensure_proj(z_cat.shape[-1], device=z_cat.device, dtype=z_cat.dtype)
        z = self.proj(z_cat)
        return z, loss_align


class HybridDLSATeacher(nn.Module):
    def __init__(self, dlsa_encoder, in_dim: int, hidden_dim: int, topk: int, noise_std: float):
        super().__init__()
        self.fast = DummyDLSAEncoder(in_dim=in_dim, hidden_dim=hidden_dim)
        self.dlsa = dlsa_encoder
        self.topk = topk
        self.noise_std = noise_std
        self.llm_proj = None

    def _ensure_llm_proj(self, in_dim: int, device, dtype):
        if self.llm_proj is None:
            self.llm_proj = nn.Linear(in_dim, ENCODER_HIDDEN).to(device=device, dtype=dtype)
    # LLM-teacher
    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        '''
        :param x:
        :param amounts:
        :param client_id:
        :param align_lambda:
        :return:
        先算 fast encoder
        只有在门控打开时，才算 LLM 语义
        用 cosine loss 拉近两者
        '''
        z_fast, _ = self.fast(x, amounts)

        if (align_lambda is None) or (align_lambda <= 0) or (self.dlsa is None):
            return z_fast, torch.tensor(0.0, device=z_fast.device, dtype=z_fast.dtype)

        x_np = x.detach().cpu().numpy()
        amt_np = amounts.detach().cpu().numpy()

        t1_list, t2_list = [], []
        for i in range(x_np.shape[0]):
            t1, t2 = _make_two_views_text(x_np[i], float(amt_np[i]), self.topk, self.noise_std)
            t1_list.append(t1)
            t2_list.append(t2)

        z_cat, _ = self.dlsa(
            t1_list, t2_list,
            client_id=client_id,
            align_lambda=float(align_lambda)
        )

        self._ensure_llm_proj(z_cat.shape[-1], device=z_cat.device, dtype=z_cat.dtype)
        z_teacher = self.llm_proj(z_cat)

        zf = F.normalize(z_fast, dim=-1)
        zt = F.normalize(z_teacher, dim=-1)
        loss_align = (1.0 - F.cosine_similarity(zf, zt, dim=-1)).mean()
        loss_align = float(align_lambda) * loss_align
        return z_fast, loss_align


class FraudHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class DLSABasedModel(nn.Module):
    def __init__(self, dlsa_encoder=None):
        super().__init__()
        self.use_llm = bool(USE_LLM_DLSA) and (dlsa_encoder is not None)
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

        self.head = FraudHead(hidden_dim=ENCODER_HIDDEN)

    def forward(self, x, amounts, client_id: int = 0, align_lambda: float = 0.0):
        z, loss_align = self.encoder(x, amounts, client_id=client_id, align_lambda=align_lambda)
        logits = self.head(z)
        return logits, z, loss_align


def _is_light_param(name: str) -> bool:
    return (
        name.startswith("head.") or
        name.startswith("encoder.mlp.") or
        name.startswith("encoder.fast.") or
        name.startswith("encoder.proj.") or
        name.startswith("encoder.llm_proj.")
    )


class Client:
    def __init__(self, client_id: int, X, y, amounts, dlsa_encoder=None):
        self.client_id = client_id
        self.dlsa_encoder = dlsa_encoder
        self.global_protos = None

        self.model = DLSABasedModel(dlsa_encoder=dlsa_encoder).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_MODEL)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(amounts, dtype=torch.float32),
        )
        self.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        self.agent = PPOAgent()
        self.memory = Memory()

        self.strategy_type = "single"
        self.ensemble_pool = None
        self._ensemble_ptr = 0
        self.w_efl = 1.0
        self.w_fpr = 1.0
        self.w_cost = 1.0

        self.threshold = 0.5

    def get_model_params(self):
        state = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in state.items() if _is_light_param(k)}

    def set_model_params(self, global_params: dict):
        state = self.model.state_dict()
        for k, v in global_params.items():
            if k in state and _is_light_param(k):
                state[k] = v.to(state[k].device)
        self.model.load_state_dict(state, strict=False)

    def set_strategy(self, strategy_type: str, weights_tuple, ensemble_pool=None):
        if strategy_type == "single":
            w1, w2, w3 = weights_tuple
            self.strategy_type = "single"
            self.ensemble_pool = None
            self.set_strategy_weights(w1, w2, w3)
        else:
            w1, w2, w3 = weights_tuple
            self.strategy_type = "ensemble"
            self.ensemble_pool = ensemble_pool
            self._ensemble_ptr = 0
            self.set_strategy_weights(w1, w2, w3)

    def set_global_prototypes(self, protos_dict):
        if protos_dict is None:
            self.global_protos = None
            return
        p0 = torch.tensor(protos_dict[0], dtype=torch.float32, device=DEVICE)
        p1 = torch.tensor(protos_dict[1], dtype=torch.float32, device=DEVICE)
        self.global_protos = {0: F.normalize(p0, dim=0), 1: F.normalize(p1, dim=0)}

    def set_strategy_weights(self, w_efl: float, w_fpr: float, w_cost: float):
        self.w_efl = float(w_efl)
        self.w_fpr = float(w_fpr)
        self.w_cost = float(w_cost)

    def _maybe_rotate_ensemble_weights(self):
        if self.strategy_type != "ensemble":
            return
        if not self.ensemble_pool:
            return
        w = self.ensemble_pool[self._ensemble_ptr % len(self.ensemble_pool)]
        self._ensemble_ptr += 1
        self.set_strategy_weights(w[0], w[1], w[2])
    # 三维状态
    def _compute_efl(self, y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> float:
        fn_mask = (y_true == 1) & (y_pred == 0)
        if fn_mask.sum() == 0:
            return 0.0
        return float(amounts[fn_mask].sum() / max(len(y_true), 1))

    def _compute_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        denom = fp + tn
        return float(fp / denom) if denom > 0 else 0.0

    def _compute_cost(self, batch_size: int) -> float:
        return float(batch_size) / 1000.0

    def _reward_transform(self, efl: float, fpr: float, cost: float) -> float:
        return - (self.w_efl * efl + self.w_fpr * fpr + self.w_cost * cost)

    def local_update(self, local_epochs=1):
        self.model.train()

        teacher = None
        # 复制一个 teacher 模型，加载当前参数，冻结梯度
        if USE_KD_DISTILL:
            teacher = DLSABasedModel(dlsa_encoder=self.dlsa_encoder).to(DEVICE)
            teacher.load_state_dict(self.model.state_dict(), strict=False)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        sum_efl = sum_fpr = sum_cost = sum_reward = 0.0
        count_batches = 0
        step = 0
        # proto_sum/proto_cnt 用来统计每类语义原型（后面给 server 聚合）
        proto_sum = {
            0: torch.zeros(ENCODER_HIDDEN, device=DEVICE),
            1: torch.zeros(ENCODER_HIDDEN, device=DEVICE),
        }
        proto_cnt = {0: 0, 1: 0}

        # ✅ 每次 local_update 视作一次 rollout：先清空 memory（更严谨）
        self.memory.clear()

        for _ in range(local_epochs):
            for X_batch, y_batch, amt_batch in self.loader:
                step += 1
                # 如果该 client 被分到 ensemble 策略，则每 batch 换一次权重
                self._maybe_rotate_ensemble_weights()

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                amt_batch = amt_batch.to(DEVICE)
                # DLSA 对齐门控：每 N 步才启用 align_lambda
                align = DLSA_ALIGN_LAMBDA if (USE_LLM_DLSA and (step % ALIGN_EVERY_N_STEPS == 0)) else 0.0

                logits, z, loss_align = self.model(
                    X_batch,
                    amounts=amt_batch,
                    client_id=self.client_id,
                    align_lambda=align
                )
                prob = torch.sigmoid(logits)
                # 监督损失：二分类 BCE
                loss_sup = F.binary_cross_entropy(prob, y_batch)
                # 原型对齐损失
                loss_proto = torch.tensor(0.0, device=DEVICE)
                if USE_PROTO_ALIGN and (self.global_protos is not None):
                    y_long = y_batch.detach().long().view(-1)
                    p0 = self.global_protos[0].unsqueeze(0)
                    p1 = self.global_protos[1].unsqueeze(0)
                    p = torch.where(y_long.unsqueeze(1) == 0, p0, p1)
                    z_n = F.normalize(z, dim=-1)
                    p_n = F.normalize(p, dim=-1)
                    loss_proto = (1.0 - F.cosine_similarity(z_n, p_n, dim=-1)).mean()
                    loss_proto = PROTO_ALIGN_LAMBDA * loss_proto
                # 蒸馏损失：student logits 对齐 teacher logits
                loss_kd = torch.tensor(0.0, device=DEVICE)
                if USE_KD_DISTILL and (teacher is not None):
                    with torch.no_grad():
                        t_logits, _, _ = teacher(
                            X_batch,
                            amounts=amt_batch,
                            client_id=self.client_id,
                            align_lambda=0.0
                        )
                    T = float(KD_T)
                    p_s = torch.sigmoid(logits / T)
                    p_t = torch.sigmoid(t_logits / T)
                    loss_kd = F.binary_cross_entropy(p_s, p_t) * (T * T)
                    loss_kd = KD_LAMBDA * loss_kd
                # 总损失
                loss = loss_sup + loss_align + loss_proto + loss_kd

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # protos
                y_long = y_batch.detach().long().view(-1)
                for cls in (0, 1):
                    m = (y_long == cls)
                    if m.any():
                        proto_sum[cls] += z[m].detach().sum(dim=0)
                        proto_cnt[cls] += int(m.sum().item())

                # reward
                y_true_np = y_batch.detach().cpu().numpy().astype(int)
                y_pred_np = (prob.detach().cpu().numpy() > self.threshold).astype(int)
                amt_np = amt_batch.detach().cpu().numpy()
                # reward 由三维风险组成，action 更新阈值。
                efl = self._compute_efl(y_true_np, y_pred_np, amt_np)
                fpr = self._compute_fpr(y_true_np, y_pred_np)
                cost = self._compute_cost(len(y_true_np))
                # 风险越大越差，所以 reward 设为负的风险加权和，PPO 的目标是最大化 reward，也就是最小化风险。
                reward = self._reward_transform(efl, fpr, cost)

                state = np.array([efl, fpr, cost], dtype=np.float32)
                action, logp = self.agent.select_action(state)
                # PPO 的 action 直接调阈值，从而改变 y_pred
                self.threshold = float(np.clip(self.threshold + 0.05 * float(action[0]), 0.01, 0.99))

                # PPO memory：补齐 dones（与 rewards 同长度）
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

        # ✅ 让最后一步成为终止（GAE 计算才严格闭合）
        if len(self.memory.dones) > 0:
            self.memory.dones[-1] = True

        self.agent.update(self.memory)
        self.memory.clear()

        if count_batches == 0:
            return {
                "client_id": self.client_id,
                "avg_efl": 0.0,
                "avg_fpr": 0.0,
                "avg_cost": 0.0,
                "avg_reward": 0.0,
                "proto_sum_0": proto_sum[0].detach().cpu().tolist(),
                "proto_sum_1": proto_sum[1].detach().cpu().tolist(),
                "proto_cnt_0": int(proto_cnt[0]),
                "proto_cnt_1": int(proto_cnt[1]),
            }

        return {
            "client_id": self.client_id,
            "avg_efl": sum_efl / count_batches,
            "avg_fpr": sum_fpr / count_batches,
            "avg_cost": sum_cost / count_batches,
            "avg_reward": sum_reward / count_batches,
            "proto_sum_0": proto_sum[0].detach().cpu().tolist(),
            "proto_sum_1": proto_sum[1].detach().cpu().tolist(),
            "proto_cnt_0": int(proto_cnt[0]),
            "proto_cnt_1": int(proto_cnt[1]),
        }
