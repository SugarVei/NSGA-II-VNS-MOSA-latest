"""
解决方案编码模块
Solution Encoding Module

使用四矩阵编码方案表示调度解决方案:
1. 机器分配矩阵 (machine_assign)
2. 序列优先级矩阵 (sequence_priority)  
3. 速度等级矩阵 (speed_level)
4. 工人技能矩阵 (worker_skill)
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from .problem import SchedulingProblem


class Solution:
    """
    调度解决方案类 - 四矩阵编码
    
    Attributes:
        machine_assign: 机器分配矩阵 [job, stage] -> machine_id
        sequence_priority: 序列优先级矩阵 [job, stage] -> priority_value
        speed_level: 速度等级矩阵 [job, stage] -> speed_level
        worker_skill: 工人技能矩阵 [job, stage] -> skill_level
        
        objectives: 目标函数值 (makespan, labor_cost, energy)
        rank: Pareto排名 (用于NSGA-II)
        crowding_distance: 拥挤度距离 (用于NSGA-II)
    """
    
    def __init__(self, 
                 n_jobs: int, 
                 n_stages: int,
                 machine_assign: Optional[np.ndarray] = None,
                 sequence_priority: Optional[np.ndarray] = None,
                 speed_level: Optional[np.ndarray] = None,
                 worker_skill: Optional[np.ndarray] = None):
        """
        初始化解决方案
        
        Args:
            n_jobs: 工件数量
            n_stages: 阶段数量
            machine_assign: 机器分配矩阵，None则创建空矩阵
            sequence_priority: 序列优先级矩阵
            speed_level: 速度等级矩阵
            worker_skill: 工人技能矩阵
        """
        self.n_jobs = n_jobs
        self.n_stages = n_stages
        
        # 四矩阵编码
        self.machine_assign = machine_assign if machine_assign is not None else np.zeros((n_jobs, n_stages), dtype=int)
        self.sequence_priority = sequence_priority if sequence_priority is not None else np.zeros((n_jobs, n_stages), dtype=int)
        self.speed_level = speed_level if speed_level is not None else np.zeros((n_jobs, n_stages), dtype=int)
        self.worker_skill = worker_skill if worker_skill is not None else np.zeros((n_jobs, n_stages), dtype=int)
        
        # 目标函数值: (makespan, labor_cost, energy)
        self.objectives: Optional[Tuple[float, float, float]] = None
        
        # NSGA-II 相关属性
        self.rank: int = 0  # Pareto排名
        self.crowding_distance: float = 0.0  # 拥挤度距离
        
        # 可行性标志
        self.is_feasible: bool = True
        self.feasibility_violations: list = []
    
    @classmethod
    def generate_random(cls, problem: 'SchedulingProblem', seed: Optional[int] = None) -> 'Solution':
        """
        根据问题定义随机生成一个可行解
        
        Args:
            problem: 调度问题实例
            seed: 随机种子
            
        Returns:
            Solution: 随机生成的解决方案
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_jobs = problem.n_jobs
        n_stages = problem.n_stages
        
        # 初始化四矩阵
        machine_assign = np.zeros((n_jobs, n_stages), dtype=int)
        sequence_priority = np.zeros((n_jobs, n_stages), dtype=int)
        speed_level = np.zeros((n_jobs, n_stages), dtype=int)
        worker_skill = np.zeros((n_jobs, n_stages), dtype=int)
        
        for job in range(n_jobs):
            for stage in range(n_stages):
                # 随机选择机器 (在该阶段可用机器范围内)
                n_machines = problem.machines_per_stage[stage]
                machine_assign[job, stage] = np.random.randint(0, n_machines)
                
                # 随机序列优先级 (用于确定同一机器上操作的顺序)
                sequence_priority[job, stage] = np.random.randint(0, 1000)
                
                # 随机速度等级
                speed = np.random.randint(0, problem.n_speed_levels)
                speed_level[job, stage] = speed
                
                # 选择可以操作该速度的最低技能工人 (降低成本)
                # 技能等级必须 >= 速度等级
                min_skill = speed
                if min_skill < problem.n_skill_levels:
                    worker_skill[job, stage] = min_skill
                else:
                    worker_skill[job, stage] = problem.n_skill_levels - 1
        
        solution = cls(
            n_jobs=n_jobs,
            n_stages=n_stages,
            machine_assign=machine_assign,
            sequence_priority=sequence_priority,
            speed_level=speed_level,
            worker_skill=worker_skill
        )
        
        return solution
    
    def copy(self) -> 'Solution':
        """创建解决方案的深拷贝"""
        new_solution = Solution(
            n_jobs=self.n_jobs,
            n_stages=self.n_stages,
            machine_assign=self.machine_assign.copy(),
            sequence_priority=self.sequence_priority.copy(),
            speed_level=self.speed_level.copy(),
            worker_skill=self.worker_skill.copy()
        )
        new_solution.objectives = self.objectives
        new_solution.rank = self.rank
        new_solution.crowding_distance = self.crowding_distance
        new_solution.is_feasible = self.is_feasible
        new_solution.feasibility_violations = self.feasibility_violations.copy()
        return new_solution
    
    def get_operation(self, job: int, stage: int) -> Tuple[int, int, int, int]:
        """
        获取指定操作的所有决策变量
        
        Returns:
            (machine, priority, speed, skill)
        """
        return (
            self.machine_assign[job, stage],
            self.sequence_priority[job, stage],
            self.speed_level[job, stage],
            self.worker_skill[job, stage]
        )
    
    def set_operation(self, job: int, stage: int, 
                      machine: int, priority: int, speed: int, skill: int):
        """设置指定操作的所有决策变量"""
        self.machine_assign[job, stage] = machine
        self.sequence_priority[job, stage] = priority
        self.speed_level[job, stage] = speed
        self.worker_skill[job, stage] = skill
        # 清除缓存的目标函数值
        self.objectives = None
    
    def dominates(self, other: 'Solution') -> bool:
        """
        检查当前解是否支配另一个解 (Pareto支配)
        
        一个解支配另一个解当且仅当:
        1. 在所有目标上都不比另一个差
        2. 在至少一个目标上严格更好
        
        注意: 所有目标都是最小化目标
        """
        if self.objectives is None or other.objectives is None:
            return False
        
        at_least_equal = True
        at_least_one_better = False
        
        for i in range(3):  # 三个目标
            if self.objectives[i] > other.objectives[i]:
                at_least_equal = False
                break
            elif self.objectives[i] < other.objectives[i]:
                at_least_one_better = True
        
        return at_least_equal and at_least_one_better
    
    def check_feasibility(self, problem: 'SchedulingProblem') -> Tuple[bool, list]:
        """
        检查解决方案的可行性
        
        Args:
            problem: 调度问题实例
            
        Returns:
            (is_feasible, violations): 是否可行及违约列表
        """
        violations = []
        
        for job in range(self.n_jobs):
            for stage in range(self.n_stages):
                machine = self.machine_assign[job, stage]
                speed = self.speed_level[job, stage]
                skill = self.worker_skill[job, stage]
                
                # 检查机器分配是否在有效范围内
                if machine >= problem.machines_per_stage[stage]:
                    violations.append(f"工件{job}阶段{stage}: 机器{machine}超出范围")
                
                # 检查速度等级是否在有效范围内
                if speed >= problem.n_speed_levels:
                    violations.append(f"工件{job}阶段{stage}: 速度{speed}超出范围")
                
                # 检查技能等级是否足够操作该速度
                if not problem.can_operate(skill, speed):
                    violations.append(f"工件{job}阶段{stage}: 技能{skill}不能操作速度{speed}")
        
        self.is_feasible = len(violations) == 0
        self.feasibility_violations = violations
        return self.is_feasible, violations
    
    def repair(self, problem: 'SchedulingProblem') -> 'Solution':
        """
        修复不可行解 (按论文规范三步修复)
        
        修复策略:
        1. 技能-速度兼容: V[i,j] > W[i,j] → 尝试提升W到最小可行水平，否则降低V
        2. 机器选择有效性: M[i,j]超出范围 → 重新随机分配
        3. 工人可用性: n_α > N_α → 降级操作的W和V
        
        Args:
            problem: 调度问题实例
            
        Returns:
            修复后的解决方案(self), 如果无法修复则标记为infeasible
        """
        self.is_feasible = True
        self.feasibility_violations = []
        
        # Step 1 & 2: 逐操作修复机器和技能-速度兼容性
        for job in range(self.n_jobs):
            for stage in range(self.n_stages):
                # 修复机器分配 (Step 2)
                n_machines = problem.machines_per_stage[stage]
                if self.machine_assign[job, stage] >= n_machines or self.machine_assign[job, stage] < 0:
                    self.machine_assign[job, stage] = np.random.randint(0, n_machines)
                
                # 修复速度等级范围
                if self.speed_level[job, stage] >= problem.n_speed_levels:
                    self.speed_level[job, stage] = problem.n_speed_levels - 1
                if self.speed_level[job, stage] < 0:
                    self.speed_level[job, stage] = 0
                
                # 修复技能等级范围
                if self.worker_skill[job, stage] >= problem.n_skill_levels:
                    self.worker_skill[job, stage] = problem.n_skill_levels - 1
                if self.worker_skill[job, stage] < 0:
                    self.worker_skill[job, stage] = 0
                
                # 修复技能-速度兼容性 (Step 1)
                speed = self.speed_level[job, stage]
                skill = self.worker_skill[job, stage]
                
                if speed > skill:  # V[i,j] > W[i,j]
                    # 尝试提升W到最小可行水平 (>= speed)
                    if speed < problem.n_skill_levels:
                        self.worker_skill[job, stage] = speed
                    else:
                        # 无法提升W，降低V到W
                        self.speed_level[job, stage] = skill
        
        # Step 3: 检查并修复工人可用性
        # 计算机器级工人分配 ω[j,f] = max{W[i,j] | M[i,j]=f}
        if problem.workers_available is not None:
            for attempt in range(3):  # 最多尝试3次修复
                machine_worker_skill = {}  # (stage, machine) -> required skill
                for job in range(self.n_jobs):
                    for stage in range(self.n_stages):
                        machine = self.machine_assign[job, stage]
                        skill = self.worker_skill[job, stage]
                        key = (stage, machine)
                        if key not in machine_worker_skill:
                            machine_worker_skill[key] = skill
                        else:
                            machine_worker_skill[key] = max(machine_worker_skill[key], skill)
                
                # 统计每个技能等级需要的工人数 n_α
                skill_count = np.zeros(problem.n_skill_levels)
                for (stage, machine), skill in machine_worker_skill.items():
                    skill_count[skill] += 1
                
                # 检查可用性约束 n_α <= N_α
                availability_ok = True
                for alpha in range(problem.n_skill_levels):
                    available = problem.workers_available[alpha] if alpha < len(problem.workers_available) else 0
                    if skill_count[alpha] > available:
                        availability_ok = False
                        # 尝试降级: 找到使用该技能的机器，降低其中一些操作的W和V
                        for job in range(self.n_jobs):
                            for stage in range(self.n_stages):
                                if self.worker_skill[job, stage] == alpha and alpha > 0:
                                    # 降级到alpha-1
                                    self.worker_skill[job, stage] = alpha - 1
                                    self.speed_level[job, stage] = min(self.speed_level[job, stage], alpha - 1)
                                    break
                            else:
                                continue
                            break
                
                if availability_ok:
                    break
            else:
                # 多次尝试后仍无法修复
                self.feasibility_violations.append("工人可用性约束无法满足")
        
        self.is_feasible = len(self.feasibility_violations) == 0
        self.objectives = None  # 清除缓存
        return self
    
    def get_makespan(self) -> float:
        """获取最大完工时间目标值"""
        return self.objectives[0] if self.objectives else float('inf')
    
    def get_labor_cost(self) -> float:
        """获取人工成本目标值"""
        return self.objectives[1] if self.objectives else float('inf')
    
    def get_energy(self) -> float:
        """获取能耗目标值"""
        return self.objectives[2] if self.objectives else float('inf')
    
    def get_weighted_sum(self, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        计算加权和 (用于标量化)
        
        Args:
            weights: 三个目标的权重 (w1, w2, w3)
            
        Returns:
            加权和值
        """
        if self.objectives is None:
            return float('inf')
        return sum(w * obj for w, obj in zip(weights, self.objectives))
    
    def __repr__(self) -> str:
        obj_str = f"({self.objectives[0]:.2f}, {self.objectives[1]:.2f}, {self.objectives[2]:.2f})" if self.objectives else "未评估"
        return f"Solution(n_jobs={self.n_jobs}, n_stages={self.n_stages}, objectives={obj_str}, rank={self.rank})"
    
    def __lt__(self, other: 'Solution') -> bool:
        """用于排序 - 先按rank排序，再按拥挤度排序"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance
