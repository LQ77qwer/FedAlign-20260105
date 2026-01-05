# server.py
"""
联邦服务器端：
    - 维护全局模型参数
    - 下发参数给各客户端
    - 聚合（FedAvg）
    - 服务器进化层：
        * multi-case fitness（client 视为 case）
        * lexicase selection（含 epsilon）
        * 相似度控制（cosine similarity）
        * individual + ensemble 共存，并让 ensemble 真正参与训练（下发成员权重池）
"""

import copy
import random
from copy import deepcopy

import numpy as np
import torch

from config import (
    DEVICE,
    POP_SIZE, EVOLVE_INTERVAL,
    ELITE_RATIO, MUTATION_STD,
    ENSEMBLE_SIZE, ENSEMBLE_RATIO,
    LEXICASE_EPS, SIM_THRESH, PARENT_TRIES
)


class Server:
    def __init__(self, clients):
        self.clients = clients
        self.num_clients = len(clients)

        # 初始化全局模型参数（拷贝第一个客户端的）
        self.global_params = copy.deepcopy(clients[0].get_model_params())

        # 策略种群
        self.population = []          # list[dict]
        self.client_assign = {}       # client_id -> strategy index（population 下标）

        self._init_population()

        # multi-case fitness 缓存：strategy_idx -> {client_id -> [reward,...]}
        self.fitness_buffer = {i: {} for i in range(len(self.population))}

    # -----------------------------
    # 初始化种群
    # -----------------------------
    def _init_population(self):
        base = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        self.population = []
        for _ in range(POP_SIZE):
            noise = np.random.normal(loc=0.0, scale=MUTATION_STD, size=3)
            w = base + noise
            w = np.maximum(w, 0.0)
            w = w / (w.sum() + 1e-8)
            self.population.append({
                "type": "single",
                "weights": w.astype(np.float32),
                "members": None,   # ensemble 时为 list[int]
            })

    # -----------------------------
    # FedAvg
    # -----------------------------
    def broadcast(self):
        for c in self.clients:
            c.set_model_params(self.global_params)

    def aggregate(self):
        """
        FedAvg：只对浮点/复数参数做 mean 聚合；
        对非浮点参数（Byte/Long/Bool 等），直接沿用原 global_params（或取第一个客户端的值）。
        """
        new_state = {}
        with torch.no_grad():
            for k, v_global in self.global_params.items():
                # 收集每个客户端该 key 的张量（都在 CPU）
                params = [c.get_model_params()[k] for c in self.clients]
                t0 = params[0]

                # ✅ 仅聚合浮点/复数
                if torch.is_floating_point(t0) or torch.is_complex(t0):
                    stacked = torch.stack([p.to(DEVICE) for p in params], dim=0)
                    new_state[k] = stacked.mean(dim=0).cpu()
                else:
                    # ✅ 非浮点：不做均值，直接保持全局（更稳）
                    # 也可以用 t0.clone()，但用 v_global 更符合“保持服务器状态”
                    new_state[k] = v_global.clone()

        self.global_params = new_state

    # -----------------------------
    # 策略下发（让 ensemble 有存在感）
    # -----------------------------
    def dispatch_strategies_to_clients(self):
        """
        将策略分配给各客户端：
            - 单体：直接下发 weights
            - 集成：下发 (平均 weights) + 成员权重池，让 client 本地轮换使用
        """
        num_strategies = len(self.population)
        for idx, c in enumerate(self.clients):
            s_idx = idx % num_strategies
            self.client_assign[c.client_id] = s_idx

            strategy = self.population[s_idx]
            if strategy["type"] == "single":
                w = strategy["weights"]
                c.set_strategy("single", (w[0], w[1], w[2]), ensemble_pool=None)
            elif strategy["type"] == "ensemble":
                members = strategy.get("members", [])
                if not members:
                    # 兜底：没有成员就当 single
                    w = strategy["weights"]
                    c.set_strategy("single", (w[0], w[1], w[2]), ensemble_pool=None)
                else:
                    member_ws = [self.population[m]["weights"] for m in members]
                    w_avg = np.mean(member_ws, axis=0)
                    w_avg = w_avg / (w_avg.sum() + 1e-8)
                    # 下发成员池：让 client 每个 batch 轮换成员权重
                    pool = [(float(wi[0]), float(wi[1]), float(wi[2])) for wi in member_ws]
                    c.set_strategy("ensemble", (w_avg[0], w_avg[1], w_avg[2]), ensemble_pool=pool)
            else:
                raise ValueError("Unknown strategy type")

    # -----------------------------
    # multi-case fitness 构造
    # -----------------------------
    def _build_case_fitness(self):
        """
        case_fitness[s_idx] = np.array(num_clients)  (每个 client=一个 case 的平均 reward)
        """
        all_client_ids = [c.client_id for c in self.clients]
        case_fitness = {}
        for s_idx, per_client_dict in self.fitness_buffer.items():
            vec = []
            for cid in all_client_ids:
                vals = per_client_dict.get(cid, [])
                if len(vals) == 0:
                    vec.append(0.0)
                else:
                    vec.append(float(np.mean(vals)))
            case_fitness[s_idx] = np.array(vec, dtype=np.float32)
        return case_fitness

    # -----------------------------
    # 相似度 & lexicase
    # -----------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return num / den

    def _lexicase_select_one(self, case_fitness: dict, candidate_ids: list, eps: float):
        """
        lexicase：随机打乱 case 顺序，逐 case 过滤候选
        eps：允许 best-eps 的候选保留，避免只剩一个
        """
        candidates = list(candidate_ids)
        case_indices = list(range(self.num_clients))
        random.shuffle(case_indices)

        for ci in case_indices:
            scores = np.array([case_fitness[sid][ci] for sid in candidates], dtype=np.float32)
            best = float(scores.max())
            keep = scores >= (best - eps)
            candidates = [sid for sid, ok in zip(candidates, keep) if ok]
            if len(candidates) <= 1:
                break

        return random.choice(candidates)

    def _select_parents_lexicase_diverse(self, case_fitness: dict, k: int,
                                        eps: float, sim_thresh: float, tries: int):
        """
        选择 k 个父代：
            - lexicase 选一个
            - 若与已选父代在 case_fitness 向量上余弦相似度过高，则重试
        """
        all_ids = list(case_fitness.keys())
        parents = []

        for _ in range(k):
            chosen = None
            for _t in range(tries):
                cand = self._lexicase_select_one(case_fitness, all_ids, eps)
                cand_vec = case_fitness[cand]

                too_sim = False
                for p in parents:
                    if self._cosine_sim(cand_vec, case_fitness[p]) >= sim_thresh:
                        too_sim = True
                        break
                if not too_sim:
                    chosen = cand
                    break

            if chosen is None:
                chosen = self._lexicase_select_one(case_fitness, all_ids, eps)

            parents.append(chosen)

        return parents

    # -----------------------------
    # 进化主入口
    # -----------------------------
    def evolve_strategies(self, round_idx: int, round_stats):
        """
        服务器进化层：
            - 根据本轮 round_stats 更新 multi-case fitness 缓存
            - 每隔 EVOLVE_INTERVAL 轮执行一次进化操作（lexicase + diversity + ensemble）
        """
        # 1) 累积本轮统计到 fitness_buffer
        for stat in round_stats:
            cid = stat["client_id"]
            reward = stat["avg_reward"]
            s_idx = self.client_assign.get(cid, None)
            if s_idx is None:
                continue
            if s_idx not in self.fitness_buffer:
                self.fitness_buffer[s_idx] = {}
            if cid not in self.fitness_buffer[s_idx]:
                self.fitness_buffer[s_idx][cid] = []
            self.fitness_buffer[s_idx][cid].append(reward)

        # 2) 不到进化轮次就返回
        if (round_idx + 1) % EVOLVE_INTERVAL != 0:
            return

        print(f"[Server] Evolution step at round {round_idx}")

        # 3) 构造 case_fitness
        case_fitness = self._build_case_fitness()

        # 4) 进化更新 population
        self._evolution_step(case_fitness)

        # 5) 清空 buffer，为下一阶段累计
        self.fitness_buffer = {i: {} for i in range(len(self.population))}

    # -----------------------------
    # 进化细节：elitism + lexicase + similarity + ensemble
    # -----------------------------
    def _evolution_step(self, case_fitness: dict):
        """
        - 精英：按 mean(case_fitness) 排序保留
        - 父代：lexicase + 相似度控制
        - 子代：
            * mutation 生成 single
            * ensemble 生成 ensemble（由若干 single 组成）
            * 如不足，补少量 crossover+mutation 的 single
        """
        num_strategies = len(self.population)
        if num_strategies == 0:
            return

        # ---- 1) 计算平均 fitness 用于精英保留 ----
        avg_fit = {sid: float(case_fitness[sid].mean()) for sid in case_fitness.keys()}
        sorted_ids = sorted(avg_fit.keys(), key=lambda x: avg_fit[x], reverse=True)

        elite_num = max(1, int(ELITE_RATIO * num_strategies))
        ensemble_num = int(round(ENSEMBLE_RATIO * num_strategies))
        ensemble_num = max(0, min(ensemble_num, num_strategies - elite_num))
        # 剩余用于 single 子代（mutation/crossover）
        remaining = num_strategies - elite_num - ensemble_num
        mutation_num = max(1, remaining)  # 先都用 mutation 生成；不够再补 crossover

        # ---- 2) 精英保留（直接拷贝）----
        new_population = []
        elite_ids = sorted_ids[:elite_num]
        for sid in elite_ids:
            new_population.append(deepcopy(self.population[sid]))

        # ---- 3) 选父代（lexicase + diversity）----
        # 需要的父代数量：mutation_num + ensemble_num * ENSEMBLE_SIZE（用于构造 ensemble）
        need_parents = mutation_num + max(0, ensemble_num * ENSEMBLE_SIZE)
        parent_ids = self._select_parents_lexicase_diverse(
            case_fitness,
            k=need_parents,
            eps=LEXICASE_EPS,
            sim_thresh=SIM_THRESH,
            tries=PARENT_TRIES
        )

        # ---- 4) mutation 生成 single 子代 ----
        def mutate_weights(w):
            noise = np.random.normal(0.0, MUTATION_STD, size=w.shape)
            child = w + noise
            child = np.maximum(child, 0.0)
            child = child / (child.sum() + 1e-8)
            return child.astype(np.float32)

        # mutation 子代
        for i in range(mutation_num):
            pid = parent_ids[i]
            base_w = self.population[pid]["weights"]
            child_w = mutate_weights(base_w)
            new_population.append({
                "type": "single",
                "weights": child_w,
                "members": None,
            })

        # ---- 5) 构造 ensemble 个体：每个 ensemble 由 ENSEMBLE_SIZE 个 single 组成 ----
        cursor = mutation_num
        for _ in range(ensemble_num):
            members = parent_ids[cursor: cursor + ENSEMBLE_SIZE]
            cursor += ENSEMBLE_SIZE
            # ensemble 自带一个平均权重（便于分析/兜底），但训练时会下发成员池
            ws = [self.population[m]["weights"] for m in members]
            w_avg = np.mean(ws, axis=0)
            w_avg = w_avg / (w_avg.sum() + 1e-8)

            new_population.append({
                "type": "ensemble",
                "members": members,
                "weights": w_avg.astype(np.float32),
            })

        # ---- 6) 若由于 rounding 导致数量不足，用 crossover+mutation 补齐 ----
        def crossover(w1, w2):
            alpha = np.random.uniform(0.0, 1.0)
            return (alpha * w1 + (1.0 - alpha) * w2).astype(np.float32)

        # 父代池（优先用精英 + 部分 lexicase 父代）
        parent_pool = list(set(elite_ids + parent_ids[:max(2, len(parent_ids) // 2)]))
        while len(new_population) < num_strategies:
            p1, p2 = np.random.choice(parent_pool, size=2, replace=True)
            w1 = self.population[p1]["weights"]
            w2 = self.population[p2]["weights"]
            child_w = mutate_weights(crossover(w1, w2))
            new_population.append({
                "type": "single",
                "weights": child_w,
                "members": None,
            })

        # ---- 7) 截断到固定大小 ----
        self.population = new_population[:num_strategies]
