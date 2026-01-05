# config.py
"""
全局配置
"""

# ===== 数据相关 =====
CSV_PATH = "default_pre.csv"   # 你的数据文件名
LABEL_COL = "labels"           # 标签列名
AMOUNT_COL = "LIMIT_BAL"       # 用来近似金额的列（可后续替换）

NUM_FEATURES = 81              # 特征维度（去掉 labels），请根据实际需要改
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
POP_SIZE = 10              # 策略个体总数（含单体和集成展开后，可先只用单体）
EVOLVE_INTERVAL = 5        # 每隔多少个 global round 触发一次进化
ELITE_RATIO = 0.2          # 精英保留比例
MUTATION_STD = 0.05        # 变异时对权重加的高斯噪声标准差

# ensemble 基础结构
ENSEMBLE_SIZE = 3          # 一个 ensemble 由多少个 single 组成（2~5都可，这里沿用你原先的 3）
ENSEMBLE_RATIO = 0.3       # 新一代种群中 ensemble 所占比例（其余是 single）

# ===== Lexicase Selection + 相似度控制 =====
LEXICASE_EPS = 0.02        # lexicase 的 epsilon 容忍（避免过拟合某一个 case）
SIM_THRESH = 0.95          # 相似度阈值（越低越鼓励多样性；0.90~0.98可调）
PARENT_TRIES = 10          # 为了避免太相似，选父代时最多重试次数

# ===== DLSA / LLM config =====
USE_LLM_DLSA = True

# LLM_DLSA_MODE:
#   - "encoder": 直接把 LLM 输出当作表征（最慢，且 TEST 也会很慢）
#   - "teacher": 默认推荐。数值 MLP 做表征，LLM 只在对齐门控开启时算对齐损失（大幅提速）
LLM_DLSA_MODE = "teacher"

# 每隔多少个 batch 才跑一次 LLM 对齐（teacher 模式下生效）
# 你每个 client 约 7477 样本，batch=64，大概 117 step/epoch
# 建议：200 => 基本 0~1 次/epoch；100 => 1 次/epoch；50 => 2 次/epoch（你现在这个）
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

