"""
解码器模块
Decoder Module

将四矩阵编码(M-Q-V-W)的解决方案解码为实际调度,并计算三个目标函数值:
1. F1: Makespan (最大完工时间)
2. F2: Labor Cost (人工成本) - 基于机器-工人分配，按标准工期计薪
3. F3: Energy Consumption (能耗) - 五类能耗累加
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from .problem import SchedulingProblem
from .solution import Solution


class Decoder:
    """
    调度解码器 - 将编码解转换为实际调度并计算目标值
    
    目标计算方式:
    - F1 (Makespan): 所有工件完成的最大时间
    - F2 (Labor Cost): 统计使用的机器，每台机器配一名工人按班次工资计算
    - F3 (Energy): 加工能耗 + 换模能耗 + 空闲能耗 + 运输能耗 + 辅助能耗
    """
    
    def __init__(self, problem: SchedulingProblem):
        """
        初始化解码器
        
        Args:
            problem: 调度问题实例
        """
        self.problem = problem
    
    def decode(self, solution: Solution) -> Tuple[float, float, float]:
        """
        解码解决方案并计算三个目标函数值
        
        Args:
            solution: 待解码的解决方案
            
        Returns:
            (makespan, labor_cost, energy): 三个目标函数值
        """
        problem = self.problem
        n_jobs = problem.n_jobs
        n_stages = problem.n_stages
        
        # ===== 时间追踪 =====
        job_completion = np.zeros((n_jobs, n_stages))  # 工件完成时间
        machine_available = defaultdict(lambda: defaultdict(float))  # 机器可用时间
        machine_last_job = defaultdict(lambda: defaultdict(lambda: -1))  # 机器上一个加工的工件
        
        # ===== 工人可用性追踪 (修复法核心) =====
        # worker_busy_until[skill] = [end_time1, end_time2, ...] 每个工人的忙碌结束时间
        workers_per_skill = problem.workers_available if problem.workers_available is not None else [5] * problem.n_skill_levels
        worker_busy_until = {skill: [0.0] * int(count) for skill, count in enumerate(workers_per_skill)}
        
        # ===== 能耗追踪 =====
        total_processing_energy = 0.0
        total_setup_energy = 0.0
        total_transport_energy = 0.0
        
        # 机器使用情况: (stage, machine) -> {skill, used, total_proc_time, total_setup_time}
        machine_usage = defaultdict(lambda: {'skill': 0, 'used': False, 
                                              'proc_time': 0.0, 'setup_time': 0.0,
                                              'segments': []})
        
        # ===== 逐阶段解码 =====
        for stage in range(n_stages):
            # 获取该阶段所有操作及优先级，按机器分组
            machine_queues = defaultdict(list)
            for job in range(n_jobs):
                machine = solution.machine_assign[job, stage]
                priority = solution.sequence_priority[job, stage]
                machine_queues[machine].append((priority, job))
            
            # 每个机器内部按优先级排序
            for machine in machine_queues:
                machine_queues[machine].sort(key=lambda x: x[0])
            
            # 按优先级顺序处理每台机器上的工件
            for machine in range(problem.machines_per_stage[stage]):
                queue = machine_queues.get(machine, [])
                
                for idx, (priority, job) in enumerate(queue):
                    speed = solution.speed_level[job, stage]
                    skill = solution.worker_skill[job, stage]
                    
                    # 获取加工时间
                    proc_time = problem.get_processing_time(job, stage, machine, speed)
                    
                    # 获取换模时间 (序列相关)
                    prev_job = machine_last_job[stage][machine]
                    setup_time = problem.get_setup_time(stage, machine, prev_job, job)
                    
                    # 计算开始时间
                    if stage == 0:
                        job_ready = 0.0
                    else:
                        # 加上运输时间
                        transport_t = problem.get_transport_time(stage - 1)
                        job_ready = job_completion[job, stage - 1] + transport_t
                        # 运输能耗
                        total_transport_energy += problem.transport_power * transport_t / 60.0
                    
                    machine_ready = machine_available[stage][machine]
                    earliest_start = max(job_ready, machine_ready)
                    
                    # ===== 修复法：检查工人可用性 =====
                    skill_idx = min(skill, len(worker_busy_until) - 1)
                    if len(worker_busy_until.get(skill_idx, [])) > 0:
                        worker_list = worker_busy_until[skill_idx]
                        # 找最早空闲的工人
                        worker_idx = 0
                        worker_available_time = worker_list[0]
                        for i, busy_until in enumerate(worker_list):
                            if busy_until <= earliest_start:
                                worker_idx = i
                                worker_available_time = busy_until
                                break
                            elif busy_until < worker_available_time:
                                worker_idx = i
                                worker_available_time = busy_until
                        
                        # 如果工人仍忙碌，延迟开始时间
                        start_time = max(earliest_start, worker_available_time)
                    else:
                        start_time = earliest_start
                        worker_idx = 0
                    
                    # 换模后开始加工
                    processing_start = start_time + setup_time
                    end_time = processing_start + proc_time
                    
                    # 更新工人忙碌时间
                    if skill_idx in worker_busy_until and len(worker_busy_until[skill_idx]) > worker_idx:
                        worker_busy_until[skill_idx][worker_idx] = end_time
                    
                    # 更新时间记录
                    job_completion[job, stage] = end_time
                    machine_available[stage][machine] = end_time
                    machine_last_job[stage][machine] = job
                    
                    # 记录机器使用情况
                    machine_usage[(stage, machine)]['used'] = True
                    machine_usage[(stage, machine)]['skill'] = max(machine_usage[(stage, machine)]['skill'], skill)
                    machine_usage[(stage, machine)]['proc_time'] += proc_time
                    machine_usage[(stage, machine)]['setup_time'] += setup_time
                    machine_usage[(stage, machine)]['segments'].append({
                        'job': job, 'start': start_time, 'setup_end': processing_start,
                        'end': end_time, 'speed': speed, 'skill': skill, 'worker_idx': worker_idx
                    })
                    
                    # 加工能耗
                    proc_power = problem.get_processing_power(stage, machine, speed)
                    total_processing_energy += proc_power * proc_time / 60.0
                    
                    # 换模能耗
                    setup_power = problem.get_setup_power(stage, machine)
                    total_setup_energy += setup_power * setup_time / 60.0
        
        # ===== 计算 F1: Makespan =====
        makespan = np.max(job_completion[:, -1]) if n_jobs > 0 else 0.0
        
        # ===== 计算 F2: Labor Cost =====
        # 每台使用的机器配一名工人，按该机器所需最高技能等级的工资计算
        total_labor_cost = 0.0
        for (stage, machine), usage in machine_usage.items():
            if usage['used']:
                skill = usage['skill']
                wage = problem.get_wage(skill)  # 标准工期工资
                total_labor_cost += wage
        
        # ===== 计算 F3: Energy (五类能耗) =====
        # 1. 加工能耗 - 已计算
        # 2. 换模能耗 - 已计算
        # 3. 空闲能耗
        total_idle_energy = 0.0
        for (stage, machine), usage in machine_usage.items():
            if usage['used']:
                # 计算空闲时间 = makespan - 加工时间 - 换模时间
                idle_time = makespan - usage['proc_time'] - usage['setup_time']
                if idle_time > 0:
                    idle_power = problem.get_idle_power(stage, machine)
                    total_idle_energy += idle_power * idle_time / 60.0
        
        # 4. 运输能耗 - 已计算
        
        # 5. 辅助能耗 (与makespan成正比)
        total_aux_energy = problem.aux_power * makespan / 60.0
        
        # 总能耗
        total_energy = (total_processing_energy + total_setup_energy + 
                        total_idle_energy + total_transport_energy + total_aux_energy)
        
        # 缓存目标函数值
        solution.objectives = (makespan, total_labor_cost, total_energy)
        
        return makespan, total_labor_cost, total_energy
    
    def decode_with_schedule(self, solution: Solution) -> Tuple[Tuple[float, float, float], Dict]:
        """
        解码解决方案并返回详细调度信息 (用于可视化甘特图)
        
        Args:
            solution: 待解码的解决方案
            
        Returns:
            (objectives, schedule_info): 目标函数值和详细调度信息
        """
        problem = self.problem
        n_jobs = problem.n_jobs
        n_stages = problem.n_stages
        
        # 调度详情
        schedule = {
            'operations': [],           # 所有工序详情
            'machine_utilization': {},  # 机器利用率
            'job_completion': {},       # 工件完成时间
            'machine_workers': {},      # 机器-工人分配
            'energy_breakdown': {},     # 能耗分解
        }
        
        # 时间和能耗追踪
        job_completion = np.zeros((n_jobs, n_stages))
        machine_available = defaultdict(lambda: defaultdict(float))
        machine_last_job = defaultdict(lambda: defaultdict(lambda: -1))
        
        # 工人可用性追踪 (修复法)
        workers_per_skill = problem.workers_available if problem.workers_available is not None else [5] * problem.n_skill_levels
        worker_busy_until = {skill: [0.0] * int(count) for skill, count in enumerate(workers_per_skill)}
        
        total_processing_energy = 0.0
        total_setup_energy = 0.0
        total_transport_energy = 0.0
        
        machine_usage = defaultdict(lambda: {'skill': 0, 'used': False, 
                                              'proc_time': 0.0, 'setup_time': 0.0})
        
        # 逐阶段解码
        for stage in range(n_stages):
            machine_queues = defaultdict(list)
            for job in range(n_jobs):
                machine = solution.machine_assign[job, stage]
                priority = solution.sequence_priority[job, stage]
                machine_queues[machine].append((priority, job))
            
            for machine in machine_queues:
                machine_queues[machine].sort(key=lambda x: x[0])
            
            for machine in range(problem.machines_per_stage[stage]):
                queue = machine_queues.get(machine, [])
                machine_ops = []
                
                for idx, (priority, job) in enumerate(queue):
                    speed = solution.speed_level[job, stage]
                    skill = solution.worker_skill[job, stage]
                    
                    proc_time = problem.get_processing_time(job, stage, machine, speed)
                    prev_job = machine_last_job[stage][machine]
                    setup_time = problem.get_setup_time(stage, machine, prev_job, job)
                    
                    if stage == 0:
                        job_ready = 0.0
                        transport_t = 0.0
                    else:
                        transport_t = problem.get_transport_time(stage - 1)
                        job_ready = job_completion[job, stage - 1] + transport_t
                        total_transport_energy += problem.transport_power * transport_t / 60.0
                    
                    machine_ready = machine_available[stage][machine]
                    earliest_start = max(job_ready, machine_ready)
                    
                    # 修复法：检查工人可用性
                    skill_idx = min(skill, len(worker_busy_until) - 1)
                    worker_idx = 0
                    if len(worker_busy_until.get(skill_idx, [])) > 0:
                        worker_list = worker_busy_until[skill_idx]
                        worker_available_time = worker_list[0]
                        for i, busy_until in enumerate(worker_list):
                            if busy_until <= earliest_start:
                                worker_idx = i
                                worker_available_time = busy_until
                                break
                            elif busy_until < worker_available_time:
                                worker_idx = i
                                worker_available_time = busy_until
                        start_time = max(earliest_start, worker_available_time)
                    else:
                        start_time = earliest_start
                    
                    processing_start = start_time + setup_time
                    end_time = processing_start + proc_time
                    
                    # 更新工人忙碌时间
                    if skill_idx in worker_busy_until and len(worker_busy_until[skill_idx]) > worker_idx:
                        worker_busy_until[skill_idx][worker_idx] = end_time
                    
                    job_completion[job, stage] = end_time
                    machine_available[stage][machine] = end_time
                    machine_last_job[stage][machine] = job
                    
                    machine_usage[(stage, machine)]['used'] = True
                    machine_usage[(stage, machine)]['skill'] = max(machine_usage[(stage, machine)]['skill'], skill)
                    machine_usage[(stage, machine)]['proc_time'] += proc_time
                    machine_usage[(stage, machine)]['setup_time'] += setup_time
                    
                    # 记录操作详情 (包含worker_idx)
                    op_info = {
                        'job': job,
                        'stage': stage,
                        'machine': machine,
                        'start': start_time,
                        'setup_end': processing_start,
                        'end': end_time,
                        'processing_time': proc_time,
                        'setup_time': setup_time,
                        'transport_time': transport_t,
                        'speed': speed,
                        'skill': skill,
                        'worker_idx': worker_idx,
                        'priority': priority
                    }
                    schedule['operations'].append(op_info)
                    machine_ops.append(op_info)
                    
                    # 能耗计算
                    proc_power = problem.get_processing_power(stage, machine, speed)
                    total_processing_energy += proc_power * proc_time / 60.0
                    setup_power = problem.get_setup_power(stage, machine)
                    total_setup_energy += setup_power * setup_time / 60.0
                
                schedule['machine_utilization'][(stage, machine)] = machine_ops
        
        # 计算目标函数
        makespan = np.max(job_completion[:, -1]) if n_jobs > 0 else 0.0
        
        # F2: Labor Cost
        total_labor_cost = 0.0
        for (stage, machine), usage in machine_usage.items():
            if usage['used']:
                skill = usage['skill']
                wage = problem.get_wage(skill)
                total_labor_cost += wage
                schedule['machine_workers'][(stage, machine)] = {'skill': skill, 'wage': wage}
        
        # F3: Energy
        total_idle_energy = 0.0
        for (stage, machine), usage in machine_usage.items():
            if usage['used']:
                idle_time = makespan - usage['proc_time'] - usage['setup_time']
                if idle_time > 0:
                    idle_power = problem.get_idle_power(stage, machine)
                    total_idle_energy += idle_power * idle_time / 60.0
        
        total_aux_energy = problem.aux_power * makespan / 60.0
        total_energy = (total_processing_energy + total_setup_energy + 
                        total_idle_energy + total_transport_energy + total_aux_energy)
        
        # 记录能耗分解
        schedule['energy_breakdown'] = {
            'processing': total_processing_energy,
            'setup': total_setup_energy,
            'idle': total_idle_energy,
            'transport': total_transport_energy,
            'auxiliary': total_aux_energy,
            'total': total_energy
        }
        
        # 记录工件完成时间
        for job in range(n_jobs):
            schedule['job_completion'][job] = job_completion[job, -1]
        
        solution.objectives = (makespan, total_labor_cost, total_energy)
        
        return (makespan, total_labor_cost, total_energy), schedule
    
    def evaluate_population(self, population: List[Solution]) -> None:
        """
        评估整个种群的目标函数值
        
        Args:
            population: 解决方案列表
        """
        for solution in population:
            if solution.objectives is None:
                self.decode(solution)


def normalize_objectives(solutions: List[Solution]) -> np.ndarray:
    """
    归一化目标函数值 (用于VNS/MOSA的标量化)
    
    Args:
        solutions: 解决方案列表
        
    Returns:
        normalized: 归一化后的目标值矩阵 [n_solutions, 3]
    """
    if not solutions:
        return np.array([])
    
    objectives = np.array([s.objectives for s in solutions if s.objectives is not None])
    
    if len(objectives) == 0:
        return np.array([])
    
    # 计算每个目标的最小值和最大值
    min_vals = objectives.min(axis=0)
    max_vals = objectives.max(axis=0)
    
    # 避免除零
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    
    # 归一化到 [0, 1]
    normalized = (objectives - min_vals) / ranges
    
    return normalized


def scalarize(objectives: Tuple[float, float, float], 
              weights: Tuple[float, float, float],
              ref_min: np.ndarray = None,
              ref_max: np.ndarray = None) -> float:
    """
    将多目标值标量化为单一综合值
    
    Args:
        objectives: 三个目标值 (F1, F2, F3)
        weights: 权重 (w1, w2, w3)
        ref_min: 参考最小值 (用于归一化)
        ref_max: 参考最大值 (用于归一化)
        
    Returns:
        综合标量值 (越小越好)
    """
    obj = np.array(objectives)
    w = np.array(weights)
    
    # 归一化权重
    w = w / (w.sum() + 1e-10)
    
    if ref_min is not None and ref_max is not None:
        # 归一化目标值
        ranges = ref_max - ref_min
        ranges[ranges == 0] = 1.0
        obj_norm = (obj - ref_min) / ranges
    else:
        obj_norm = obj
    
    return float(np.dot(w, obj_norm))
