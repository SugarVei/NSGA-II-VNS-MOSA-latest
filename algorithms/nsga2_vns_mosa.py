# -*- coding: utf-8 -*-
"""
NSGA-II-VNS-MOSA 混合多目标优化算法
Hybrid NSGA-II with Variable Neighborhood Search and Multi-Objective Simulated Annealing

本文件是论文《基于四矩阵编码的混合流水车间调度优化研究》的核心算法实现。

算法核心思想：
1. NSGA-II 框架负责全局进化搜索（非支配排序 + 拥挤度距离）
2. VNS 变邻域搜索在每一代进化后对精英个体进行局部增强
3. MOSA 模拟退火准则引入概率接受机制，维持搜索多样性

基于四矩阵编码 (M-Q-V-W)：
- M (Machine Assignment): 机器分配矩阵
- Q (Sequence Priority): 工序优先级矩阵
- V (Speed Level): 加工速度等级矩阵
- W (Worker Skill): 工人技能等级矩阵

作者: [您的姓名]
日期: 2024
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import random
import math
from copy import deepcopy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder


class NSGA2_VNS_MOSA:
    """
    NSGA-II-VNS-MOSA 混合多目标优化算法
    
    融合了三种优化策略：
    1. NSGA-II: 非支配排序遗传算法，提供全局搜索能力
    2. VNS: 变邻域搜索，提供局部精化能力
    3. MOSA: 多目标模拟退火，提供跳出局部最优的能力
    
    Attributes:
        problem: 调度问题实例
        pop_size: 种群大小
        n_generations: 进化代数
        vns_max_iters: VNS 每次搜索的最大迭代次数
        initial_temp: 模拟退火初始温度
        cooling_rate: 降温速率
        elite_ratio: 每代进行 VNS 增强的精英个体比例
    """
    
    def __init__(self,
                 problem: SchedulingProblem,
                 pop_size: int = 50,
                 n_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 vns_max_iters: int = 20,
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.95,
                 elite_ratio: float = 0.2,
                 seed: Optional[int] = None):
        """
        初始化 NSGA-II-VNS-MOSA 算法
        
        Args:
            problem: 调度问题实例
            pop_size: 种群大小
            n_generations: 进化代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            vns_max_iters: VNS 最大迭代次数
            initial_temp: 模拟退火初始温度
            cooling_rate: 降温速率
            elite_ratio: 精英个体比例（用于 VNS 增强）
            seed: 随机种子
        """
        self.problem = problem
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.vns_max_iters = vns_max_iters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.elite_ratio = elite_ratio
        
        self.decoder = Decoder(problem)
        self.population: List[Solution] = []
        self.temperature = initial_temp
        
        # 收敛历史
        self.convergence_history = {
            'generation': [],
            'best_makespan': [],
            'best_labor_cost': [],
            'best_energy': [],
            'n_pareto': [],
            'temperature': []
        }
        
        # 进度回调
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    # ==================== NSGA-II 核心组件 ====================
    
    def initialize_population(self) -> List[Solution]:
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            sol = Solution.generate_random(self.problem)
            sol.repair(self.problem)
            self.decoder.decode(sol)
            population.append(sol)
        return population
    
    def non_dominated_sort(self, population: List[Solution]) -> List[List[int]]:
        """非支配排序"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        if not fronts[0]:
            for i in range(n):
                population[i].rank = 0
                fronts[0].append(i)
            return fronts
        
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return [f for f in fronts if f]
    
    def calculate_crowding_distance(self, population: List[Solution], front: List[int]):
        """计算拥挤度距离"""
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = float('inf')
            return
        
        for i in front:
            population[i].crowding_distance = 0.0
        
        for obj_idx in range(3):
            sorted_front = sorted(front, key=lambda x: population[x].objectives[obj_idx])
            population[sorted_front[0]].crowding_distance = float('inf')
            population[sorted_front[-1]].crowding_distance = float('inf')
            
            obj_min = population[sorted_front[0]].objectives[obj_idx]
            obj_max = population[sorted_front[-1]].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            for k in range(1, len(sorted_front) - 1):
                curr_idx = sorted_front[k]
                prev_idx = sorted_front[k - 1]
                next_idx = sorted_front[k + 1]
                distance = (population[next_idx].objectives[obj_idx] - 
                           population[prev_idx].objectives[obj_idx]) / obj_range
                population[curr_idx].crowding_distance += distance
    
    def tournament_selection(self, population: List[Solution]) -> Solution:
        """锦标赛选择"""
        idx1, idx2 = random.sample(range(len(population)), 2)
        sol1, sol2 = population[idx1], population[idx2]
        
        if sol1.rank < sol2.rank:
            return sol1.copy()
        elif sol2.rank < sol1.rank:
            return sol2.copy()
        elif sol1.crowding_distance > sol2.crowding_distance:
            return sol1.copy()
        else:
            return sol2.copy()
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """交叉操作 - 工件级别交叉"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        n_jobs = self.problem.n_jobs
        crossover_point = random.randint(1, n_jobs - 1)
        
        for job in range(crossover_point, n_jobs):
            for stage in range(self.problem.n_stages):
                child1.machine_assign[job, stage], child2.machine_assign[job, stage] = \
                    child2.machine_assign[job, stage], child1.machine_assign[job, stage]
                child1.sequence_priority[job, stage], child2.sequence_priority[job, stage] = \
                    child2.sequence_priority[job, stage], child1.sequence_priority[job, stage]
                child1.speed_level[job, stage], child2.speed_level[job, stage] = \
                    child2.speed_level[job, stage], child1.speed_level[job, stage]
                child1.worker_skill[job, stage], child2.worker_skill[job, stage] = \
                    child2.worker_skill[job, stage], child1.worker_skill[job, stage]
        
        child1.objectives = None
        child2.objectives = None
        return child1, child2
    
    def mutate(self, solution: Solution) -> Solution:
        """变异操作"""
        if random.random() > self.mutation_prob:
            return solution
        
        mutation_type = random.choice(['operation', 'job', 'sequence'])
        n_jobs, n_stages = self.problem.n_jobs, self.problem.n_stages
        
        if mutation_type == 'operation':
            job = random.randint(0, n_jobs - 1)
            stage = random.randint(0, n_stages - 1)
            n_machines = self.problem.machines_per_stage[stage]
            solution.machine_assign[job, stage] = random.randint(0, n_machines - 1)
            new_speed = random.randint(0, self.problem.n_speed_levels - 1)
            solution.speed_level[job, stage] = new_speed
            solution.worker_skill[job, stage] = new_speed
        elif mutation_type == 'job':
            job = random.randint(0, n_jobs - 1)
            for stage in range(n_stages):
                n_machines = self.problem.machines_per_stage[stage]
                solution.machine_assign[job, stage] = random.randint(0, n_machines - 1)
                new_speed = random.randint(0, self.problem.n_speed_levels - 1)
                solution.speed_level[job, stage] = new_speed
                solution.worker_skill[job, stage] = new_speed
        else:
            stage = random.randint(0, n_stages - 1)
            if n_jobs >= 2:
                job1, job2 = random.sample(range(n_jobs), 2)
                solution.sequence_priority[job1, stage], solution.sequence_priority[job2, stage] = \
                    solution.sequence_priority[job2, stage], solution.sequence_priority[job1, stage]
        
        solution.objectives = None
        return solution
    
    # ==================== VNS 变邻域搜索组件 ====================
    
    def vns_neighborhood_N1(self, sol: Solution) -> Solution:
        """N1: 机器重新分配"""
        neighbor = sol.copy()
        job = random.randint(0, self.problem.n_jobs - 1)
        stage = random.randint(0, self.problem.n_stages - 1)
        n_machines = self.problem.machines_per_stage[stage]
        if n_machines > 1:
            current_m = neighbor.machine_assign[job, stage]
            new_m = current_m
            while new_m == current_m:
                new_m = random.randint(0, n_machines - 1)
            neighbor.machine_assign[job, stage] = new_m
        neighbor.objectives = None
        return neighbor
    
    def vns_neighborhood_N2(self, sol: Solution) -> Solution:
        """N2: 工序优先级调整"""
        neighbor = sol.copy()
        job = random.randint(0, self.problem.n_jobs - 1)
        stage = random.randint(0, self.problem.n_stages - 1)
        neighbor.sequence_priority[job, stage] = random.randint(0, 1000)
        neighbor.objectives = None
        return neighbor
    
    def vns_neighborhood_N3(self, sol: Solution) -> Solution:
        """N3: 速度-技能同步变异"""
        neighbor = sol.copy()
        job = random.randint(0, self.problem.n_jobs - 1)
        stage = random.randint(0, self.problem.n_stages - 1)
        n_speeds = self.problem.n_speed_levels
        if n_speeds > 1:
            curr_v = neighbor.speed_level[job, stage]
            new_v = (curr_v + random.choice([-1, 1])) % n_speeds
            neighbor.speed_level[job, stage] = new_v
            if neighbor.worker_skill[job, stage] < new_v:
                neighbor.worker_skill[job, stage] = new_v
        neighbor.objectives = None
        return neighbor
    
    def vns_neighborhood_N4(self, sol: Solution) -> Solution:
        """N4: 阶段工件交换"""
        neighbor = sol.copy()
        stage = random.randint(0, self.problem.n_stages - 1)
        if self.problem.n_jobs >= 2:
            j1, j2 = random.sample(range(self.problem.n_jobs), 2)
            neighbor.machine_assign[j1, stage], neighbor.machine_assign[j2, stage] = \
                neighbor.machine_assign[j2, stage], neighbor.machine_assign[j1, stage]
            neighbor.sequence_priority[j1, stage], neighbor.sequence_priority[j2, stage] = \
                neighbor.sequence_priority[j2, stage], neighbor.sequence_priority[j1, stage]
            neighbor.speed_level[j1, stage], neighbor.speed_level[j2, stage] = \
                neighbor.speed_level[j2, stage], neighbor.speed_level[j1, stage]
            neighbor.worker_skill[j1, stage], neighbor.worker_skill[j2, stage] = \
                neighbor.worker_skill[j2, stage], neighbor.worker_skill[j1, stage]
        neighbor.objectives = None
        return neighbor
    
    def vns_neighborhood_N5(self, sol: Solution) -> Solution:
        """N5: 工件级全决策重塑 (Shaking)"""
        neighbor = sol.copy()
        job = random.randint(0, self.problem.n_jobs - 1)
        for s in range(self.problem.n_stages):
            n_m = self.problem.machines_per_stage[s]
            neighbor.machine_assign[job, s] = random.randint(0, n_m - 1)
            neighbor.sequence_priority[job, s] = random.randint(0, 1000)
            v = random.randint(0, self.problem.n_speed_levels - 1)
            neighbor.speed_level[job, s] = v
            neighbor.worker_skill[job, s] = random.randint(v, self.problem.n_skill_levels - 1)
        neighbor.objectives = None
        return neighbor
    
    def vns_search(self, solution: Solution) -> Solution:
        """执行 VNS 搜索"""
        current = solution.copy()
        if current.objectives is None:
            self.decoder.decode(current)
        
        neighborhoods = [
            self.vns_neighborhood_N1,
            self.vns_neighborhood_N2,
            self.vns_neighborhood_N3,
            self.vns_neighborhood_N4,
            self.vns_neighborhood_N5
        ]
        
        for _ in range(self.vns_max_iters):
            k = 0
            while k < len(neighborhoods):
                neighbor = neighborhoods[k](current)
                neighbor.repair(self.problem)
                self.decoder.decode(neighbor)
                
                # 使用 MOSA 准则判断是否接受
                if self._mosa_accept(current, neighbor):
                    current = neighbor
                    k = 0
                else:
                    k += 1
        
        return current
    
    # ==================== MOSA 模拟退火组件 ====================
    
    def _mosa_accept(self, current: Solution, neighbor: Solution) -> bool:
        """MOSA 接受准则"""
        if neighbor.dominates(current):
            return True
        elif current.dominates(neighbor):
            d1 = neighbor.objectives[0] - current.objectives[0]
            d2 = neighbor.objectives[1] - current.objectives[1]
            d3 = neighbor.objectives[2] - current.objectives[2]
            delta = (max(0, d1) + max(0, d2) + max(0, d3)) / 3.0
            prob = math.exp(-delta / self.temperature) if self.temperature > 0.1 else 0
            return random.random() < prob
        else:
            return random.random() < 0.7
    
    # ==================== 主运行流程 ====================
    
    def run(self) -> List[Solution]:
        """
        运行 NSGA-II-VNS-MOSA 混合算法
        
        Returns:
            Pareto 最优解集
        """
        if self.progress_callback:
            self.progress_callback(0, self.n_generations, "正在初始化种群...")
        
        self.population = self.initialize_population()
        self.temperature = self.initial_temp
        
        for gen in range(self.n_generations):
            if self.progress_callback:
                self.progress_callback(gen + 1, self.n_generations,
                                       f"NSGA-II-VNS-MOSA 第 {gen + 1}/{self.n_generations} 代")
            
            # 1. 非支配排序
            fronts = self.non_dominated_sort(self.population)
            
            # 2. 计算拥挤度
            for front in fronts:
                self.calculate_crowding_distance(self.population, front)
            
            # 3. VNS 增强精英个体
            if fronts:
                elite_count = max(1, int(len(fronts[0]) * self.elite_ratio))
                for idx in fronts[0][:elite_count]:
                    self.population[idx] = self.vns_search(self.population[idx])
            
            # 4. 记录收敛历史
            self._record_convergence(gen, fronts)
            
            # 5. 创建子代
            offspring = self._create_offspring()
            
            # 6. 合并并选择下一代
            combined = self.population + offspring
            self.population = self._select_next_generation(combined)
            
            # 7. 降温
            self.temperature *= self.cooling_rate
        
        # 最终非支配排序
        fronts = self.non_dominated_sort(self.population)
        pareto_solutions = [self.population[i] for i in fronts[0]]
        
        if self.progress_callback:
            self.progress_callback(self.n_generations, self.n_generations,
                                   f"算法完成，找到 {len(pareto_solutions)} 个 Pareto 解")
        
        return pareto_solutions
    
    def _create_offspring(self) -> List[Solution]:
        """创建子代"""
        offspring = []
        while len(offspring) < self.pop_size:
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            child1.repair(self.problem)
            child2.repair(self.problem)
            self.decoder.decode(child1)
            self.decoder.decode(child2)
            offspring.extend([child1, child2])
        return offspring[:self.pop_size]
    
    def _select_next_generation(self, combined: List[Solution]) -> List[Solution]:
        """选择下一代"""
        fronts = self.non_dominated_sort(combined)
        for front in fronts:
            self.calculate_crowding_distance(combined, front)
        
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                for i in front:
                    next_gen.append(combined[i])
            else:
                remaining = self.pop_size - len(next_gen)
                sorted_front = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                for i in sorted_front[:remaining]:
                    next_gen.append(combined[i])
                break
        return next_gen
    
    def _record_convergence(self, gen: int, fronts: List[List[int]]):
        """记录收敛历史"""
        objectives = np.array([s.objectives for s in self.population])
        self.convergence_history['generation'].append(gen)
        self.convergence_history['best_makespan'].append(objectives[:, 0].min())
        self.convergence_history['best_labor_cost'].append(objectives[:, 1].min())
        self.convergence_history['best_energy'].append(objectives[:, 2].min())
        self.convergence_history['n_pareto'].append(len(fronts[0]) if fronts else 0)
        self.convergence_history['temperature'].append(self.temperature)
    
    def get_convergence_data(self) -> dict:
        """获取收敛历史"""
        return self.convergence_history


# ==================== 独立测试入口 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("NSGA-II-VNS-MOSA 混合算法测试")
    print("=" * 60)
    
    # 创建测试问题
    problem = SchedulingProblem.generate_random(
        n_jobs=20, n_stages=3, machines_per_stage=[3, 2, 4],
        n_speed_levels=3, n_skill_levels=3, seed=42
    )
    
    print(f"问题规模: {problem.n_jobs} 工件 × {problem.n_stages} 阶段")
    print(f"机器配置: {problem.machines_per_stage}")
    
    # 运行算法
    algorithm = NSGA2_VNS_MOSA(
        problem,
        pop_size=50,
        n_generations=50,
        vns_max_iters=10,
        seed=42
    )
    
    def progress(current, total, msg):
        print(f"  [{current}/{total}] {msg}")
    
    algorithm.set_progress_callback(progress)
    
    print("\n开始运行算法...")
    pareto_front = algorithm.run()
    
    print(f"\n找到 {len(pareto_front)} 个 Pareto 最优解:")
    print("-" * 50)
    print(f"{'解编号':^8} {'Makespan':^12} {'人工成本':^12} {'能耗':^12}")
    print("-" * 50)
    for i, sol in enumerate(pareto_front[:10]):  # 显示前10个
        print(f"{i+1:^8} {sol.objectives[0]:^12.2f} {sol.objectives[1]:^12.2f} {sol.objectives[2]:^12.2f}")
    
    if len(pareto_front) > 10:
        print(f"  ... 共 {len(pareto_front)} 个解")
    
    print("\n测试完成!")
