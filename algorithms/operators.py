# -*- coding: utf-8 -*-
"""
统一算子库
Unified Operators Library

为所有多目标优化算法提供基于四矩阵编码的交叉、变异和邻域生成算子。
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from copy import deepcopy
import random

if TYPE_CHECKING:
    from models.problem import SchedulingProblem
    from models.solution import Solution
    from models.decoder import Decoder


def four_matrix_sx_crossover(
    parent1: 'Solution',
    parent2: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> Tuple['Solution', 'Solution']:
    """
    四矩阵交换序列交叉 (SX Crossover)
    
    交叉策略：
    - 对每个阶段：根据 sequence_priority 得到两个父代的排序序列
    - 构造 swap_path：从 order1 变换到 order2 所需的 swap 序列
    - 沿 swap_path 逐步生成候选解（同步交换四矩阵）
    - 从候选解中选择最优的两个作为子代
    
    Args:
        parent1: 父代1
        parent2: 父代2
        rng: numpy 随机数生成器
        problem: 调度问题实例
        decoder: 解码器
        
    Returns:
        (child1, child2): 两个子代解
    """
    n_jobs = problem.n_jobs
    n_stages = problem.n_stages
    
    # 工件级交叉：随机选择交叉点
    crossover_point = rng.integers(1, n_jobs) if n_jobs > 1 else 1
    
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # 交换 crossover_point 之后的工件的四矩阵值
    for job in range(crossover_point, n_jobs):
        for stage in range(n_stages):
            # 交换四个矩阵的对应元素
            child1.machine_assign[job, stage], child2.machine_assign[job, stage] = \
                child2.machine_assign[job, stage], child1.machine_assign[job, stage]
            
            child1.sequence_priority[job, stage], child2.sequence_priority[job, stage] = \
                child2.sequence_priority[job, stage], child1.sequence_priority[job, stage]
            
            child1.speed_level[job, stage], child2.speed_level[job, stage] = \
                child2.speed_level[job, stage], child1.speed_level[job, stage]
            
            child1.worker_skill[job, stage], child2.worker_skill[job, stage] = \
                child2.worker_skill[job, stage], child1.worker_skill[job, stage]
    
    # 清除缓存的目标值
    child1.objectives = None
    child2.objectives = None
    
    # 修复并评估
    child1.repair(problem)
    child2.repair(problem)
    decoder.decode(child1)
    decoder.decode(child2)
    
    return child1, child2


def mutation_single(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem'
) -> 'Solution':
    """
    单点变异
    
    随机选择一个 (job, stage) 位置，然后随机选择四矩阵中的一个进行变异。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    job = rng.integers(0, problem.n_jobs)
    stage = rng.integers(0, problem.n_stages)
    
    # 随机选择变异类型
    mutation_type = rng.choice(['machine', 'priority', 'speed', 'skill'])
    
    if mutation_type == 'machine':
        n_machines = problem.machines_per_stage[stage]
        if n_machines > 1:
            current = mutant.machine_assign[job, stage]
            new_machine = rng.integers(0, n_machines)
            while new_machine == current and n_machines > 1:
                new_machine = rng.integers(0, n_machines)
            mutant.machine_assign[job, stage] = new_machine
            
    elif mutation_type == 'priority':
        # 与另一个随机工件交换优先级
        if problem.n_jobs > 1:
            other_job = rng.integers(0, problem.n_jobs)
            while other_job == job:
                other_job = rng.integers(0, problem.n_jobs)
            mutant.sequence_priority[job, stage], mutant.sequence_priority[other_job, stage] = \
                mutant.sequence_priority[other_job, stage], mutant.sequence_priority[job, stage]
                
    elif mutation_type == 'speed':
        current = mutant.speed_level[job, stage]
        delta = rng.choice([-1, 1])
        new_speed = max(0, min(problem.n_speed_levels - 1, current + delta))
        mutant.speed_level[job, stage] = new_speed
        
    else:  # skill
        current = mutant.worker_skill[job, stage]
        delta = rng.choice([-1, 1])
        new_skill = max(0, min(problem.n_skill_levels - 1, current + delta))
        mutant.worker_skill[job, stage] = new_skill
    
    mutant.objectives = None
    mutant.repair(problem)
    
    return mutant


def mutation_multi(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    k: int = 3
) -> 'Solution':
    """
    多点变异
    
    连续执行 k 次单点变异。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        k: 变异次数
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    for _ in range(k):
        mutant = mutation_single(mutant, rng, problem)
    
    return mutant


def mutation_inversion(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem'
) -> 'Solution':
    """
    反转变异
    
    对某个阶段的序列进行片段反转，同步应用到四矩阵。
    
    Args:
        solution: 待变异的解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        
    Returns:
        变异后的解（已修复，未评估）
    """
    mutant = solution.copy()
    
    stage = rng.integers(0, problem.n_stages)
    n_jobs = problem.n_jobs
    
    if n_jobs < 3:
        return mutant
    
    # 按当前 sequence_priority 得到排序
    priorities = mutant.sequence_priority[:, stage]
    order = np.argsort(priorities)
    
    # 随机选择反转区间 [i, j]
    i = rng.integers(0, n_jobs - 1)
    j = rng.integers(i + 1, n_jobs)
    
    # 反转区间
    reversed_segment = order[i:j+1][::-1]
    new_order = np.concatenate([order[:i], reversed_segment, order[j+1:]])
    
    # 根据新顺序重新赋值 priority
    for rank, job_idx in enumerate(new_order):
        mutant.sequence_priority[job_idx, stage] = rank
    
    mutant.objectives = None
    mutant.repair(problem)
    
    return mutant


def simple_neighbor(
    solution: 'Solution',
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    mode: Optional[str] = None
) -> 'Solution':
    """
    简单邻域生成
    
    为 MOSA(use_vns=False) 提供邻域候选。随机选择一种变异方式生成邻居。
    
    Args:
        solution: 当前解
        rng: numpy 随机数生成器
        problem: 调度问题实例
        mode: 变异模式 ('single', 'multi', 'inversion')，None 则随机选择
        
    Returns:
        邻域解（已修复，未评估）
    """
    if mode is None:
        mode = rng.choice(['single', 'multi', 'inversion'])
    
    if mode == 'single':
        return mutation_single(solution, rng, problem)
    elif mode == 'multi':
        return mutation_multi(solution, rng, problem, k=3)
    else:  # inversion
        return mutation_inversion(solution, rng, problem)


def apply_crossover_with_probability(
    parent1: 'Solution',
    parent2: 'Solution',
    crossover_prob: float,
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> Tuple['Solution', 'Solution']:
    """
    带概率的交叉操作
    
    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_prob: 交叉概率
        rng: numpy 随机数生成器
        problem: 调度问题实例
        decoder: 解码器
        
    Returns:
        (child1, child2): 两个子代
    """
    if rng.random() > crossover_prob:
        c1 = parent1.copy()
        c2 = parent2.copy()
        if c1.objectives is None:
            decoder.decode(c1)
        if c2.objectives is None:
            decoder.decode(c2)
        return c1, c2
    
    return four_matrix_sx_crossover(parent1, parent2, rng, problem, decoder)


def apply_mutation_with_probability(
    solution: 'Solution',
    mutation_prob: float,
    rng: np.random.Generator,
    problem: 'SchedulingProblem',
    decoder: 'Decoder'
) -> 'Solution':
    """
    带概率的变异操作
    
    Args:
        solution: 待变异的解
        mutation_prob: 变异概率
        rng: numpy 随机数生成器
        problem: 调度问题实例
        decoder: 解码器
        
    Returns:
        变异后的解
    """
    if rng.random() > mutation_prob:
        return solution
    
    # 随机选择变异类型
    mutation_type = rng.choice(['single', 'multi', 'inversion'])
    
    if mutation_type == 'single':
        mutant = mutation_single(solution, rng, problem)
    elif mutation_type == 'multi':
        mutant = mutation_multi(solution, rng, problem)
    else:
        mutant = mutation_inversion(solution, rng, problem)
    
    decoder.decode(mutant)
    return mutant
