# config.py
"""
全局配置
"""

# ===== 数据相关 =====
CSV_PATH = "creditcard_augmented.csv"   # 你的数据文件名
LABEL_COL = "isFradulent"           # 标签列名
AMOUNT_COL = "Transaction_amount"       # 用来近似金额的列（可后续替换）

NUM_FEATURES = 8              # 特征维度（去掉 labels），请根据实际需要改
NUM_CLIENTS = 5                # 模拟的机构数
TEST_RATIO = 0.2               # 将一部分数据留作测试（后面可扩展）

# ===== 联邦训练相关 =====
GLOBAL_ROUNDS = 20             # 联邦通信轮数
LOCAL_EPOCHS = 1               # 每一轮中每个客户端本地训练的 epoch 数
BATCH_SIZE = 64               # 本地 batch size

# ===== 模型结构 =====
ENCODER_HIDDEN = 128           # DLSA 占位 encoder 的输出维度
HEAD_HIDDEN = 64               # 如需再加层可用，目前直接线性到 1

# ===== 优化器超参（分类模型） =====
LR_MODEL = 1e-3
WEIGHT_DECAY = 1e-5

# ===== PPO & 强化学习相关 =====
STATE_DIM = 3                  # 状态 = [r_efl, r_fpr, r_cost]
ACTION_DIM = 1                 # 单一动作：阈值微调量 delta
LR_PPO = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4

# ===== 风险度量归一化 & 权重 =====
MAX_EFL = 1.0                # 经验上限，用于归一化
MAX_FPR = 1.0
MAX_COST = 1.0

W_EFL = 0.4                    # 三个风险在 scalar reward 中的权重
W_FPR = 0.4
W_COST = 0.2

# ===== 设备 =====
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 服务器进化层相关 =====
POP_SIZE = 10              # 策略个体总数
EVOLVE_INTERVAL = 5        # 每隔多少个 global round 触发一次进化
ELITE_RATIO = 0.2          # 精英保留比例
MUTATION_STD = 0.05        # 变异时对权重加的高斯噪声标准差

# ensemble 基础结构
ENSEMBLE_SIZE = 3          # 一个 ensemble 由多少个 single 组成
ENSEMBLE_RATIO = 0.3       # 新一代种群中 ensemble 所占比例

# ===== Lexicase Selection + 相似度控制 =====
LEXICASE_EPS = 0.02        # lexicase 的 epsilon 容忍
SIM_THRESH = 0.95          # 相似度阈值（越低越鼓励多样性）
PARENT_TRIES = 10          # 选父代最多重试次数

# ===== DLSA / LLM config =====
USE_LLM_DLSA = True

# LLM_DLSA_MODE:
#   - "encoder": 直接把 LLM 输出当作表征（最慢，且 TEST 也会很慢）
#   - "teacher": 默认推荐。数值 MLP 做表征，LLM 只在对齐门控开启时算对齐损失（大幅提速）
LLM_DLSA_MODE = "teacher"

# 每隔多少个 batch 才跑一次 LLM 对齐（teacher 模式下生效）
ALIGN_EVERY_N_STEPS = 100

LLM_MODEL_ID = "Qwen/Qwen3-4B"
LLM_MAX_LEN = 128
LLM_4BIT = True
LLM_BATCH = 8
LLM_EMB_POOLING = "last_token"  # or "mean"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

DLSA_ALIGN_LAMBDA = 0.1
DLSA_PROMPT_TOPK = 12
DLSA_VIEW_NOISE = 0.05


# ===== Prototype Alignment (全局原型对齐) =====
USE_PROTO_ALIGN = True
PROTO_ALIGN_LAMBDA = 0.05      # 原型对齐损失权重(0.01~0.1)
PROTO_MOMENTUM = 0.9           # 服务器更新原型的动量(0~0.99)，0 表示直接替换

# ===== Semantic Distillation (语义蒸馏/KD) =====
USE_KD_DISTILL = True
KD_LAMBDA = 0.2                # 蒸馏损失权重(0.05~0.5)
KD_T = 2.0                     # temperature
