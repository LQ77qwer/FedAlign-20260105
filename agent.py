# agent.py
"""
PPO 强化学习 Agent，实现阈值自适应策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ====== 可选依赖：允许在未安装 transformers/peft/bitsandbytes 的环境下跑通 Dummy 路径 ======
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    _HAS_LLM = True
    _LLM_IMPORT_ERROR = None
except Exception as _e:  # noqa
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = None
    LoraConfig = get_peft_model = None
    _HAS_LLM = False
    _LLM_IMPORT_ERROR = _e

from config import (
    STATE_DIM, ACTION_DIM, LR_PPO, GAMMA, GAE_LAMBDA,
    CLIP_EPS, PPO_EPOCHS, DEVICE
)


# ================= PPO 网络 =================

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))  # action mean in [-1, 1]


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []   # ✅ 必须与 rewards 同长度

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        self.policy = PolicyNet(state_dim, action_dim).to(DEVICE)
        self.value = ValueNet(state_dim).to(DEVICE)

        self.policy_old = PolicyNet(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=LR_PPO
        )

    @torch.no_grad()
    def select_action(self, state_np):
        # policy_old(state) 输出 action mean（tanh 到 [-1,1]）
        # 采样 Normal(mean, std=0.3)，并记录 logprob
        state = torch.tensor(state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mean = self.policy_old(state)
        std = 0.3
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).cpu().numpy(), float(logprob.item())

    def update(self, memory: Memory):
        """
        使用 memory 里的 trajectory 做 PPO 更新
        先做一个非常重要的鲁棒性修复：保证 dones 与 rewards 长度一致（避免 GAE 崩）
        然后取 states/actions/old_logprobs/value，做 GAE advantages（后续继续看下去就是 PPO clip 更新）
        """
        if len(memory.states) == 0:
            return

        rewards = memory.rewards
        dones = memory.dones

        # ✅ 关键兜底：确保 dones 与 rewards 同长度
        if (dones is None) or (len(dones) != len(rewards)):
            dones = [0.0 for _ in range(len(rewards))]
            if len(dones) > 0:
                dones[-1] = 1.0  # 终止一次 rollout
            memory.dones = dones

        states = torch.tensor(np.array(memory.states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32, device=DEVICE)
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32, device=DEVICE)

        values = self.value(states).squeeze(-1).detach().cpu().numpy().tolist()

        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = np.array(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + np.array(values, dtype=np.float32)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        for _ in range(PPO_EPOCHS):
            mean = self.policy(states)
            std = 0.3
            dist = torch.distributions.Normal(mean, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            value_pred = self.value(states).squeeze(-1)
            loss_value = F.mse_loss(value_pred, returns)

            loss_entropy = -entropy.mean()

            loss = loss_policy + 0.5 * loss_value + 0.01 * loss_entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


# ================= DLSA Encoder（LLM + 双LoRA） =================

class DLSAEncoder(nn.Module):
    """
    DLSA: Domain-Level Semantic Alignment encoder
    - shared LoRA (task adapter): Ag
    - private LoRA per client (domain adapter): As_i
    """

    def __init__(
        self,
        model_id=None,
        max_len=None,
        use_4bit=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=None,
        pooling="last_token",
    ):
        super().__init__()
        if not _HAS_LLM:
            raise RuntimeError(
                f"DLSAEncoder 需要 transformers/peft/bitsandbytes 等依赖，但当前不可用：{_LLM_IMPORT_ERROR}"
            )

        from config import (
            LLM_MODEL_ID, LLM_MAX_LEN, LLM_4BIT, LORA_R, LORA_ALPHA, LORA_DROPOUT,
            LORA_TARGET_MODULES, LLM_EMB_POOLING
        )

        self.model_id = model_id or LLM_MODEL_ID
        self.max_len = int(max_len or LLM_MAX_LEN)
        self.use_4bit = bool(use_4bit if use_4bit is not None else LLM_4BIT)
        self.lora_r = int(lora_r or LORA_R)
        self.lora_alpha = int(lora_alpha or LORA_ALPHA)
        self.lora_dropout = float(lora_dropout if lora_dropout is not None else LORA_DROPOUT)
        self.target_modules = target_modules or LORA_TARGET_MODULES
        self.pooling = pooling or LLM_EMB_POOLING

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = None
        if self.use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            quantization_config=quant_cfg,
            device_map="auto" if self.use_4bit else None,
        )

        lora_cfg_shared = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        #
        self.model_shared = get_peft_model(base, lora_cfg_shared)
        self.domain_models = {}

    def _ensure_domain_adapter(self, client_id: int):
        name = f"domain_{client_id}"
        if name in getattr(self.model_shared, "peft_config", {}):
            return name

        lora_cfg_private = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model_shared.add_adapter(name, lora_cfg_private)
        return name

    def _get_backbone(self, model):
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()
        else:
            base = model

        if hasattr(base, "model"):
            return base.model
        if hasattr(base, "transformer"):
            return base.transformer
        if hasattr(base, "backbone"):
            return base.backbone
        return base

    @torch.no_grad()
    def _encode_text(self, model, texts):
        from config import LLM_BATCH

        device = next(model.parameters()).device
        backbone = self._get_backbone(model)

        outs = []
        for i in range(0, len(texts), LLM_BATCH):
            chunk = texts[i:i + LLM_BATCH]

            enc = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = backbone(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            hs = out.hidden_states[-1] if (
                hasattr(out, "hidden_states") and out.hidden_states is not None
            ) else out.last_hidden_state

            if self.pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                idx = enc["attention_mask"].sum(dim=1) - 1
                pooled = hs[torch.arange(hs.size(0), device=hs.device), idx, :]

            outs.append(pooled)

        return torch.cat(outs, dim=0)

    def forward(self, texts_view1, texts_view2, client_id: int = 0, align_lambda: float = 0.0):
        m = self.model_shared

        if not (align_lambda and align_lambda > 0):
            try:
                m.set_adapter("default")
            except Exception:
                pass
            z1 = self._encode_text(m, texts_view1)
            z = torch.cat([z1, z1], dim=-1)
            loss_align = torch.tensor(0.0, device=z.device)
            return z, loss_align

        try:
            m.set_adapter("default")
        except Exception:
            pass
        z1 = self._encode_text(m, texts_view1)

        domain_name = self._ensure_domain_adapter(client_id)
        m.set_adapter(domain_name)
        z2 = self._encode_text(m, texts_view2)

        z = torch.cat([z1, z2], dim=-1)

        z1n = F.normalize(z1, dim=-1)
        z2n = F.normalize(z2, dim=-1)
        loss_align = (1.0 - (z1n * z2n).sum(dim=-1)).mean() * float(align_lambda)
        return z, loss_align
