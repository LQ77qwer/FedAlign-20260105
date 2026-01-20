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
    - ✅ 新增：
        * Prototype Alignment：聚合全局原型并下发给客户端
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
    LEXICASE_EPS, SIM_THRESH, PARENT_TRIES,
    ENCODER_HIDDEN, USE_PROTO_ALIGN, PROTO_MOMENTUM
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
        # 一个策略不是一个标量 fitness，而是一个 跨客户端的性能向量。
        self.fitness_buffer = {i: {} for i in range(len(self.population))}

        # 全局原型（Prototype Alignment）：{0: np.ndarray[H], 1: np.ndarray[H]}
        self.global_protos = None

    # -----------------------------
    # 初始化种群
    # -----------------------------
    def _init_population(self):
        base = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        self.population = []
        for _ in range(POP_SIZE):
            noise = np.random.normal(loc=0.0, scale=MUTATION_STD, size=3)
            w = base + noise
            w = np.clip(w, 1e-6, None)
            w = w / w.sum()
            self.population.append({"type": "single", "w": w.astype(np.float32)})

        self._dispatch_clients()

    # -----------------------------
    # client 分配策略
    # -----------------------------
    def _dispatch_clients(self):
        self.client_assign = {}
        for cid in range(self.num_clients):
            self.client_assign[cid] = cid % len(self.population)

        for c in self.clients:
            idx = self.client_assign[c.client_id]
            indiv = self.population[idx]
            if indiv["type"] == "single":
                w = indiv["w"]
                c.set_strategy("single", (float(w[0]), float(w[1]), float(w[2])), ensemble_pool=None)
            else:
                pool = indiv["pool"]   # list[np.ndarray]
                wavg = np.mean(np.stack(pool, axis=0), axis=0)
                pool_list = [(float(p[0]), float(p[1]), float(p[2])) for p in pool]
                c.set_strategy("ensemble", (float(wavg[0]), float(wavg[1]), float(wavg[2])), ensemble_pool=pool_list)

    # -----------------------------
    # 广播与聚合
    # -----------------------------
    def broadcast(self):
        for c in self.clients:
            c.set_model_params(self.global_params)

    # ----- Prototype Alignment: broadcast global prototypes to clients -----
    def broadcast_prototypes(self):
        if not USE_PROTO_ALIGN:
            return
        for c in self.clients:
            if hasattr(c, "set_global_prototypes"):
                c.set_global_prototypes(self.global_protos)

    # ----- Prototype Alignment: aggregate prototypes from round_stats -----
    def aggregate_prototypes(self, round_stats):
        if not USE_PROTO_ALIGN:
            return
        sum0 = np.zeros((ENCODER_HIDDEN,), dtype=np.float32)
        sum1 = np.zeros((ENCODER_HIDDEN,), dtype=np.float32)
        cnt0 = 0
        cnt1 = 0
        for s in round_stats:
            if ("proto_sum_0" not in s) or ("proto_sum_1" not in s):
                continue
            c0 = int(s.get("proto_cnt_0", 0))
            c1 = int(s.get("proto_cnt_1", 0))
            if c0 > 0:
                sum0 += np.array(s["proto_sum_0"], dtype=np.float32)
                cnt0 += c0
            if c1 > 0:
                sum1 += np.array(s["proto_sum_1"], dtype=np.float32)
                cnt1 += c1

        if cnt0 == 0 or cnt1 == 0:
            return

        p0 = sum0 / (cnt0 + 1e-8)
        p1 = sum1 / (cnt1 + 1e-8)

        # normalize
        p0 = p0 / (np.linalg.norm(p0) + 1e-8)
        p1 = p1 / (np.linalg.norm(p1) + 1e-8)

        if (self.global_protos is None) or (PROTO_MOMENTUM is None) or (PROTO_MOMENTUM <= 0):
            self.global_protos = {0: p0, 1: p1}
        else:
            m = float(PROTO_MOMENTUM)
            self.global_protos[0] = m * self.global_protos[0] + (1 - m) * p0
            self.global_protos[1] = m * self.global_protos[1] + (1 - m) * p1

    def aggregate(self):
        """
        FedAvg：只对浮点/复数参数做 mean 聚合；
        对非浮点参数（Byte/Long/Bool 等），直接沿用原 global_params（或取第一个客户端的值）。
        """
        new_state = {}
        with torch.no_grad():
            for k, v_global in self.global_params.items():
                vals = []
                dtypes = []
                for c in self.clients:
                    local_state = c.get_model_params()
                    if k in local_state:
                        vals.append(local_state[k].to(DEVICE))
                        dtypes.append(vals[-1].dtype)

                if len(vals) == 0:
                    new_state[k] = v_global
                    continue

                if not torch.is_floating_point(vals[0]) and not torch.is_complex(vals[0]):
                    new_state[k] = vals[0].detach().cpu()
                    continue

                stacked = torch.stack([v.to(dtype=vals[0].dtype) for v in vals], dim=0)
                mean_v = stacked.mean(dim=0)
                new_state[k] = mean_v.detach().cpu()

        self.global_params = new_state

    # -----------------------------
    # Evolution（保留你原来的实现）
    # -----------------------------
    def evolve_strategies(self, round_stats, round_idx: int):
        # 更新 fitness buffer
        for s in round_stats:
            cid = int(s["client_id"])
            idx = int(self.client_assign.get(cid, 0))
            if idx not in self.fitness_buffer:
                self.fitness_buffer[idx] = {}
            if cid not in self.fitness_buffer[idx]:
                self.fitness_buffer[idx][cid] = []
            self.fitness_buffer[idx][cid].append(float(s["avg_reward"]))

        if (round_idx + 1) % EVOLVE_INTERVAL != 0:
            return

        self._evolution_step()
        self._dispatch_clients()

    # ===== 以下进化逻辑（与你当前版本一致）=====
    def _build_case_fitness(self):
        case_fitness = []
        for i in range(len(self.population)):
            vec = []
            for cid in range(self.num_clients):
                lst = self.fitness_buffer.get(i, {}).get(cid, [])
                vec.append(float(np.mean(lst)) if len(lst) > 0 else -1e9)
            case_fitness.append(np.array(vec, dtype=np.float32))
        return case_fitness

    def _lexicase_select(self, candidates, case_order, case_fitness):
        pool = candidates[:]
        for cid in case_order:
            scores = [case_fitness[i][cid] for i in pool]
            best = np.max(scores)
            eps = LEXICASE_EPS
            pool = [i for i in pool if case_fitness[i][cid] >= best - eps]
            if len(pool) <= 1:
                break
        return random.choice(pool)

    def _cos_sim(self, a: np.ndarray, b: np.ndarray):
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))

    def _select_parents_lexicase_diverse(self, case_fitness):
        candidates = list(range(len(self.population)))
        case_order = list(range(self.num_clients))
        random.shuffle(case_order)

        p1 = self._lexicase_select(candidates, case_order, case_fitness)

        tries = 0
        while True:
            tries += 1
            random.shuffle(case_order)
            p2 = self._lexicase_select(candidates, case_order, case_fitness)

            if p2 == p1:
                if tries >= PARENT_TRIES:
                    break
                continue

            sim = self._cos_sim(case_fitness[p1], case_fitness[p2])
            if sim < SIM_THRESH:
                break
            if tries >= PARENT_TRIES:
                break
        return p1, p2

    def _mutate_w(self, w: np.ndarray):
        noise = np.random.normal(0.0, MUTATION_STD, size=w.shape)
        w2 = w + noise
        w2 = np.clip(w2, 1e-6, None)
        w2 = w2 / w2.sum()
        return w2.astype(np.float32)

    def _crossover(self, w1: np.ndarray, w2: np.ndarray):
        alpha = np.random.uniform(0.0, 1.0)
        w = alpha * w1 + (1 - alpha) * w2
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
        return w.astype(np.float32)

    def _evolution_step(self):
        case_fitness = self._build_case_fitness()

        # elite
        mean_fit = [float(np.mean(cf)) for cf in case_fitness]
        elite_n = max(1, int(len(self.population) * ELITE_RATIO))
        elite_idx = np.argsort(mean_fit)[::-1][:elite_n]
        new_pop = [deepcopy(self.population[i]) for i in elite_idx]

        target_ens = int(POP_SIZE * ENSEMBLE_RATIO)
        target_single = POP_SIZE - target_ens

        # fill singles
        while len([p for p in new_pop if p["type"] == "single"]) < target_single:
            p1, p2 = self._select_parents_lexicase_diverse(case_fitness)
            w = self._crossover(self.population[p1]["w"], self.population[p2]["w"])
            w = self._mutate_w(w)
            new_pop.append({"type": "single", "w": w})

        # fill ensembles
        while len([p for p in new_pop if p["type"] == "ensemble"]) < target_ens:
            members = []
            for _ in range(ENSEMBLE_SIZE):
                p1, _ = self._select_parents_lexicase_diverse(case_fitness)
                members.append(deepcopy(self.population[p1]["w"]))
            new_pop.append({"type": "ensemble", "pool": members})

        self.population = new_pop[:POP_SIZE]
        self.fitness_buffer = {i: {} for i in range(len(self.population))}
