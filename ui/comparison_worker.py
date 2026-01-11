# -*- coding: utf-8 -*-
"""
算法对比试验后台工作线程
Comparison Worker Thread

在后台执行多算例×多算法对比实验，避免阻塞 UI。
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from PyQt5.QtCore import QThread, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.problem import SchedulingProblem
from models.solution import Solution
from algorithms.nsga2 import NSGAII
from algorithms.mosa import MOSA
from algorithms.moead import MOEAD
from algorithms.spea2 import SPEA2
from algorithms.mopso import MOPSO
from algorithms.hybrid_variants import NSGA2_VNS, NSGA2_MOSA
from algorithms.nsga2_vns_mosa import NSGA2_VNS_MOSA  # 独立的核心算法文件
from experiments.taguchi.pareto import build_pf_ref
from experiments.taguchi.metrics import compute_all_metrics, get_normalization_info

# 导入 CaseConfig
from ui.case_data import CaseConfig

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def run_algorithm_task(args):
    """
    全局函数，用于在子进程中运行单个算法任务。
    必须定义在顶层以支持 pickle 序列化。
    """
    case_no = args['case_no']
    run_idx = args['run_idx']
    alg_name = args['alg_name']
    case = args['case']
    params = args['params']
    alg_seed = args['seed']
    base_seed = args['base_seed']

    # 1. 创建问题实例 (在子进程中创建以避免复杂对象传递)
    # 逻辑参考 _create_problem_from_case
    if case.is_configured and case.problem_data:
        pd = case.problem_data
        workers = pd.get('workers_available_arr')
        if workers is None:
            workers = np.array(case.workers_available)
        
        problem = SchedulingProblem(
            n_jobs=case.n_jobs,
            n_stages=3,
            machines_per_stage=case.machines_per_stage,
            n_speed_levels=3,
            n_skill_levels=3,
            processing_time=pd.get('processing_time'),
            setup_time=pd.get('setup_time'),
            transport_time=pd.get('transport_time'),
            processing_power=pd.get('processing_power'),
            setup_power=pd.get('setup_power'),
            idle_power=pd.get('idle_power'),
            transport_power=pd.get('transport_power', 0.5),
            aux_power=pd.get('aux_power', 1.0),
            skill_wages=pd.get('skill_wages'),
            workers_available=workers
        )
    else:
        seed = base_seed + case_no * 10000 + run_idx * 100
        problem = SchedulingProblem.generate_random(
            n_jobs=case.n_jobs,
            n_stages=3,
            machines_per_stage=case.machines_per_stage,
            n_speed_levels=3,
            n_skill_levels=3,
            seed=seed
        )
        problem.workers_available = np.array(case.workers_available)

    # 2. 运行算法
    objectives = np.array([])
    try:
        if alg_name == 'NSGA-II':
            alg = NSGAII(problem, pop_size=params.get('pop_size', 200),
                         n_generations=params.get('n_generations', 500), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'MOSA':
            # 标准独立的 MOSA，增加迭代次数以提高解质量
            mosa = MOSA(problem, initial_temp=params.get('initial_temp', 100.0), 
                        max_iterations=params.get('max_iterations', 500), seed=alg_seed)
            pf = mosa.run()
        elif alg_name == 'MOEA/D':
            alg = MOEAD(problem, pop_size=params.get('pop_size', 200), n_generations=params.get('n_generations', 500), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'SPEA2':
            alg = SPEA2(problem, pop_size=params.get('pop_size', 200), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'MOPSO':
            alg = MOPSO(problem, swarm_size=params.get('swarm_size', 200), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'NSGA2-VNS':
            alg = NSGA2_VNS(problem, pop_size=params.get('pop_size', 200), 
                            n_generations=params.get('n_generations', 500), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'NSGA2-MOSA':
            alg = NSGA2_MOSA(problem, pop_size=params.get('pop_size', 200), 
                             n_generations=params.get('n_generations', 500), seed=alg_seed)
            pf = alg.run()
        elif alg_name == 'NSGA2-VNS-MOSA':
            alg = NSGA2_VNS_MOSA(problem, pop_size=params.get('pop_size', 200), 
                                 n_generations=params.get('n_generations', 500), seed=alg_seed)
            pf = alg.run()
        else:
            return {'error': f"未知算法: {alg_name}", 'case_no': case_no, 'run_idx': run_idx, 'alg_name': alg_name}
        
        # 3. 提取目标值
        objectives = np.array([s.objectives for s in pf if s.objectives is not None])
    except Exception as e:
        return {'error': str(e), 'case_no': case_no, 'run_idx': run_idx, 'alg_name': alg_name}

    return {
        'case_no': case_no,
        'run_idx': run_idx,
        'alg_name': alg_name,
        'objectives': objectives,
        'error': None
    }


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}时{minutes}分"


class ComparisonWorker(QThread):
    """
    算法对比试验工作线程
    
    支持多算例 × 多算法的批量对比实验。
    
    Signals:
        progress: (current, total, message) - 进度更新
        detailed_progress: (info_dict) - 详细进度信息
        log: (message) - 日志消息
        finished_result: (results_dict) - 完成时发射结果
        error: (error_message) - 错误消息
    """
    
    progress = pyqtSignal(int, int, str)
    detailed_progress = pyqtSignal(dict)  # 详细进度信息
    log = pyqtSignal(str)
    finished_result = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    # 算法类映射
    ALGORITHM_CLASSES = {
        'NSGA-II': NSGAII,
        'MOSA': None,  # 特殊处理：需要先运行 NSGA-II
        'MOEA/D': MOEAD,
        'SPEA2': SPEA2,
        'MOPSO': MOPSO,
        'NSGA2-VNS': NSGA2_VNS,
        'NSGA2-MOSA': NSGA2_MOSA,
        'NSGA2-VNS-MOSA': NSGA2_VNS_MOSA,
    }
    
    def __init__(self,
                 selected_algorithms: List[str],
                 cases_config: List[CaseConfig],
                 params_dict: Dict[str, Dict[str, Any]],
                 runs: int = 30,
                 base_seed: int = 42,
                 weights: tuple = (0.4, 0.3, 0.3)):
        """
        初始化工作线程
        
        Args:
            selected_algorithms: 选中的算法名称列表
            cases_config: 算例配置列表
            params_dict: 全局算法参数字典（当算例未配置时使用）
            runs: 每个算例的重复次数
            base_seed: 基准随机种子
            weights: 三个目标的权重 (w1, w2, w3)，用于计算综合值
        """
        super().__init__()
        self.selected_algorithms = selected_algorithms
        self.cases_config = cases_config
        self.params_dict = params_dict
        self.runs = runs
        self.base_seed = base_seed
        self.weights = weights  # 目标权重
        self._is_cancelled = False
        
        # 时间追踪
        self._start_time = None
        self._task_times = []  # 记录每个任务的耗时
    
    def cancel(self):
        """取消执行"""
        self._is_cancelled = True
    
    def _estimate_remaining_time(self, current: int, total: int) -> float:
        """估算剩余时间（秒）"""
        if not self._task_times or current == 0:
            return 0.0
        avg_time = sum(self._task_times) / len(self._task_times)
        remaining_tasks = total - current
        return avg_time * remaining_tasks
    
    def _emit_detailed_progress(self, current: int, total: int, 
                                 case_no: int, case_scale: str,
                                 alg_name: str, run_idx: int,
                                 task_start_time: float = None):
        """发送详细进度信息"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        remaining = self._estimate_remaining_time(current, total)
        
        # 计算当前任务耗时
        task_elapsed = 0
        if task_start_time:
            task_elapsed = time.time() - task_start_time
        
        info = {
            'current': current,
            'total': total,
            'percent': (current / total * 100) if total > 0 else 0,
            'case_no': case_no,
            'case_scale': case_scale,
            'algorithm': alg_name,
            'run_idx': run_idx + 1,
            'runs_total': self.runs,
            'elapsed_time': elapsed,
            'elapsed_str': format_time(elapsed),
            'remaining_time': remaining,
            'remaining_str': format_time(remaining) if remaining > 0 else '计算中...',
            'task_elapsed': task_elapsed,
            'n_cases': len(self.cases_config),
            'n_algorithms': len(self.selected_algorithms),
        }
        self.detailed_progress.emit(info)
    
    def _create_problem_from_case(self, case: CaseConfig, run_idx: int) -> SchedulingProblem:
        """
        根据算例配置创建问题实例
        
        Args:
            case: 算例配置
            run_idx: 运行索引（用于生成不同的随机实例）
            
        Returns:
            SchedulingProblem 实例
        """
        # 如果算例已配置数据，使用配置的数据
        if case.is_configured and case.problem_data:
            pd = case.problem_data
            
            # 处理 workers_available 数组
            workers = pd.get('workers_available_arr')
            if workers is None:
                workers = np.array(case.workers_available)
            
            problem = SchedulingProblem(
                n_jobs=case.n_jobs,
                n_stages=3,
                machines_per_stage=case.machines_per_stage,
                n_speed_levels=3,
                n_skill_levels=3,
                processing_time=pd.get('processing_time'),
                setup_time=pd.get('setup_time'),
                transport_time=pd.get('transport_time'),
                processing_power=pd.get('processing_power'),
                setup_power=pd.get('setup_power'),
                idle_power=pd.get('idle_power'),
                transport_power=pd.get('transport_power', 0.5),
                aux_power=pd.get('aux_power', 1.0),
                skill_wages=pd.get('skill_wages'),
                workers_available=workers
            )
        else:
            # 未配置数据时，自动生成随机实例
            # 使用 case_no 和 run_idx 生成唯一种子
            seed = self.base_seed + case.case_no * 10000 + run_idx * 100
            
            problem = SchedulingProblem.generate_random(
                n_jobs=case.n_jobs,
                n_stages=3,
                machines_per_stage=case.machines_per_stage,
                n_speed_levels=3,
                n_skill_levels=3,
                seed=seed
            )
            
            # 设置工人数量
            problem.workers_available = np.array(case.workers_available)
        
        return problem
    
    def _get_algorithm_params(self, case: CaseConfig, alg_name: str) -> Dict[str, Any]:
        """
        获取算法参数
        
        Args:
            case: 算例配置
            alg_name: 算法名称
            
        Returns:
            算法参数字典
        """
        # 优先使用算例配置的参数
        if case.algorithm_params and alg_name in case.algorithm_params:
            return case.algorithm_params[alg_name]
        # 否则使用全局参数
        return self.params_dict.get(alg_name, {})
    
    def _run_algorithm(self, alg_name: str, problem: SchedulingProblem, 
                       params: Dict[str, Any], seed: int) -> List[Solution]:
        """运行单个算法"""
        
        if alg_name == 'NSGA-II':
            alg = NSGAII(
                problem,
                pop_size=params.get('pop_size', 200),
                n_generations=params.get('n_generations', 200),     # 对比算法减少迭代
                crossover_prob=params.get('crossover_prob', 0.95),
                mutation_prob=params.get('mutation_prob', 0.15),
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'MOSA':
            # 标准独立的 MOSA 算法 - 增加迭代次数以提高解质量 (解决 HV=0 问题)
            max_iter = params.get('max_iterations', params.get('markov_chain_length', 500))
            mosa = MOSA(
                problem,
                initial_temp=params.get('initial_temp', 100.0),
                cooling_rate=params.get('cooling_rate', 0.95),
                final_temp=params.get('final_temp', 0.001),
                max_iterations=max_iter,
                seed=seed
            )
            return mosa.run()
        
        elif alg_name == 'MOEA/D':
            alg = MOEAD(
                problem,
                pop_size=params.get('pop_size', 200),
                n_generations=params.get('n_generations', 500),     # 增加代数以提高解质量 (解决 HV=0 问题)
                neighborhood_size=params.get('neighborhood_size', 40),
                crossover_prob=params.get('crossover_prob', 0.95),
                mutation_prob=params.get('mutation_prob', 0.15),
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'SPEA2':
            alg = SPEA2(
                problem,
                pop_size=params.get('pop_size', 200),
                archive_size=params.get('archive_size', 100),
                n_generations=params.get('n_generations', 200),     # 对比算法减少迭代
                crossover_prob=params.get('crossover_prob', 0.95),
                mutation_prob=params.get('mutation_prob', 0.15),
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'MOPSO':
            alg = MOPSO(
                problem,
                swarm_size=params.get('swarm_size', 200),
                max_iterations=params.get('max_iterations', 200),   # 对比算法减少迭代
                w=params.get('w', 0.5),
                c1=params.get('c1', 1.5),
                c2=params.get('c2', 1.5),
                repository_size=params.get('repository_size', 200),
                mutation_prob=params.get('mutation_prob', 0.1),
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'NSGA2-VNS':
            vns_iters = params.get('vns_iterations', params.get('vns_neighborhood_structures', 4))
            alg = NSGA2_VNS(
                problem,
                pop_size=params.get('pop_size', 200),
                n_generations=params.get('n_generations', 200),     # 对比算法减少迭代
                crossover_prob=params.get('crossover_prob', 0.95),
                mutation_prob=params.get('mutation_prob', 0.15),
                vns_iterations=vns_iters,
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'NSGA2-MOSA':
            mosa_iters = params.get('mosa_iterations', params.get('markov_chain_length', 100))
            alg = NSGA2_MOSA(
                problem,
                pop_size=params.get('pop_size', 200),
                n_generations=params.get('n_generations', 200),     # 对比算法减少迭代
                crossover_prob=params.get('crossover_prob', 0.95),
                mutation_prob=params.get('mutation_prob', 0.15),
                initial_temp=params.get('initial_temp', 1000.0),
                cooling_rate=params.get('cooling_rate', 0.95),
                max_iterations=mosa_iters,
                seed=seed
            )
            return alg.run()
        
        elif alg_name == 'NSGA2-VNS-MOSA':
            # 兼容新参数名
            mosa_iters = params.get('mosa_iterations', params.get('markov_chain_length', 100))
            vns_iters = params.get('vns_iterations', params.get('vns_neighborhood_structures', 4))
            alg = NSGA2_VNS_MOSA(
                problem,
                pop_size=params.get('pop_size', 200),
                n_generations=params.get('n_generations', 500),
                crossover_prob=params.get('crossover_prob', 0.95),  # 田口最优: Pc=0.95
                mutation_prob=params.get('mutation_prob', 0.15),    # 田口最优: Pm=0.15
                initial_temp=params.get('initial_temp', 1000.0),    # 田口最优: T0=1000
                cooling_rate=params.get('cooling_rate', 0.95),      # 田口最优: α=0.95
                max_iterations=mosa_iters,
                vns_iterations=vns_iters,
                seed=seed
            )
            return alg.run()
        
        else:
            raise ValueError(f"未知算法: {alg_name}")
    
    def run(self):
        """执行多进程并行的多算例×多算法对比试验"""
        try:
            self._start_time = time.time()
            self._task_times = []
            
            n_cases = len(self.cases_config)
            n_algorithms = len(self.selected_algorithms)
            total_tasks = n_cases * n_algorithms * self.runs
            
            self.log.emit(f"总运行次数: {total_tasks}")
            self.log.emit(f"🚀 启动并行对比试验 (核心数: {multiprocessing.cpu_count()}):")
            self.log.emit(f"  算例数: {n_cases}, 算法数: {n_algorithms}, 每算例重复: {self.runs} 次")
            self.log.emit(f"  总运行计划: {total_tasks} 次算法运行")
            self.log.emit(f"{'─'*60}")
            
            # 准备所有任务参数
            task_args = []
            for case in self.cases_config:
                for run_idx in range(self.runs):
                    for alg_name in self.selected_algorithms:
                        params = self._get_algorithm_params(case, alg_name)
                        alg_seed = self.base_seed + case.case_no * 1000 + run_idx
                        task_args.append({
                            'case_no': case.case_no,
                            'case_scale': case.problem_scale_str,
                            'run_idx': run_idx,
                            'alg_name': alg_name,
                            'case': case,
                            'params': params,
                            'seed': alg_seed,
                            'base_seed': self.base_seed
                        })

            # 结果存储结构
            # case_all_objectives[case_no][alg_name] = [objs1, objs2, ...]
            case_all_objectives = {c.case_no: {alg: [] for alg in self.selected_algorithms} for c in self.cases_config}
            
            # 进度跟踪结构：记录每个(case_no, alg_name)组合已完成的运行次数
            progress_tracker = {c.case_no: {alg: 0 for alg in self.selected_algorithms} for c in self.cases_config}
            
            current_completed = 0
            
            # 使用进程池执行任务
            # Windows 限制: ProcessPoolExecutor 的 max_workers 不能超过 61
            max_workers = min(multiprocessing.cpu_count(), 61)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(run_algorithm_task, arg): arg for arg in task_args}
                
                # 任务提交完成后立即发送进度通知
                self.log.emit(f"✅ 所有 {total_tasks} 个任务已提交到进程池，正在等待结果...")
                self.log.emit(f"⏳ 使用 {max_workers} 个并行进程执行任务")
                self.log.emit(f"💡 提示：单个 NSGA2-VNS-MOSA 任务 (500代) 可能需要数分钟，请耐心等待...")
                self.progress.emit(0, total_tasks, f"已提交 {total_tasks} 个任务，等待第一个结果...")
                
                for future in as_completed(future_to_task):
                    if self._is_cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.log.emit("🛑 试验已中途取消")
                        return
                    
                    try:
                        res = future.result()
                        current_completed += 1
                        task_start_time = self._start_time # 简化处理，因为是并行的
                        
                        case_no = res['case_no']
                        alg_name = res['alg_name']
                        run_idx = res['run_idx']
                        
                        # 更新进度计数
                        progress_tracker[case_no][alg_name] += 1
                        alg_run_count = progress_tracker[case_no][alg_name]
                        
                        # 找到对应的算例规模字符串
                        case_scale = next((arg['case_scale'] for arg in task_args if arg['case_no'] == case_no), "Unknown")
                        
                        if res['error']:
                            self.log.emit(f"  ⚠️ [{alg_name}] Case {case_no} 第{run_idx+1}次 - 错误: {res['error']}")
                            case_all_objectives[case_no][alg_name].append(np.array([]).reshape(0, 3))
                        else:
                            n_solutions = len(res['objectives']) if len(res['objectives']) > 0 else 0
                            case_all_objectives[case_no][alg_name].append(res['objectives'])
                            
                            # 发送详细的任务完成日志
                            self.log.emit(
                                f"  ✅ [{alg_name}] Case {case_no} ({case_scale}) "
                                f"第{run_idx+1}/{self.runs}次完成 | "
                                f"Pareto解: {n_solutions}个 | "
                                f"进度: {alg_run_count}/{self.runs}"
                            )
                        
                        # 当某个算法在某个算例上的所有运行都完成时，发送汇总信息
                        if alg_run_count == self.runs:
                            self.log.emit(
                                f"  📊 [{alg_name}] Case {case_no} 全部 {self.runs} 次运行完毕"
                            )
                        
                        # 每完成一个任务更新一次进度
                        self.progress.emit(current_completed, total_tasks, f"完成 {current_completed}/{total_tasks}")
                        
                        # 发送详细进度到UI
                        self._emit_detailed_progress(
                            current_completed, total_tasks,
                            case_no, case_scale,
                            alg_name, run_idx,
                            time.time() # 这里传个假值，UI层有实时计时器了
                        )
                        
                        # 记录任务时间用于剩余时间估算 (由于并行，这里的估算会比单线程复杂，暂用平均时间/核心数)
                        elapsed = time.time() - self._start_time
                        avg_task_time = (elapsed * multiprocessing.cpu_count()) / current_completed
                        self._task_times = [avg_task_time] # 更新估算基础

                    except Exception as e:
                        self.log.emit(f"  ❌ 进程执行异常: {str(e)}")

            # --- 后处理：计算指标 ---
            self.log.emit("\n📊 所有算法运行完毕，正在计算性能指标...")
            results = {}
            
            for case in self.cases_config:
                case_no = case.case_no
                self.log.emit(f"  正在汇总 Case {case_no}...")
                
                # 构建该算例的全局参考前沿
                all_objectives_flat = []
                for alg_name in self.selected_algorithms:
                    for obj_array in case_all_objectives[case_no][alg_name]:
                        if len(obj_array) > 0:
                            all_objectives_flat.append(obj_array)
                
                if not all_objectives_flat:
                    results[case_no] = {alg: {'igd_mean': float('inf'), 'hv_mean': 0.0, 'gd_mean': float('inf'), 'n_valid_runs': 0} for alg in self.selected_algorithms}
                    continue
                
                pf_ref = build_pf_ref(all_objectives_flat)
                norm_info = get_normalization_info(pf_ref)
                f_min, f_max = np.array(norm_info['f_min']), np.array(norm_info['f_max'])
                hv_ref_point = np.array(norm_info['hv_ref_point'])
                
                results[case_no] = {}
                weights_arr = np.array(self.weights)  # 转换为numpy数组用于计算
                for alg_name in self.selected_algorithms:
                    igd_values, hv_values, gd_values, composite_values = [], [], [], []
                    for obj_array in case_all_objectives[case_no][alg_name]:
                        if len(obj_array) > 0:
                            m = compute_all_metrics(obj_array, pf_ref, f_min, f_max, hv_ref_point)
                            igd_values.append(m['igd'])
                            hv_values.append(m['hv'])
                            gd_values.append(m['gd'])
                            # 计算综合值：每个解的加权和，取最小值作为本次运行的代表值
                            composite_per_solution = obj_array @ weights_arr
                            composite_values.append(np.min(composite_per_solution))
                    
                    if igd_values:
                        results[case_no][alg_name] = {
                            'igd_mean': np.mean(igd_values), 'igd_std': np.std(igd_values, ddof=1) if len(igd_values) > 1 else 0.0,
                            'hv_mean': np.mean(hv_values), 'hv_std': np.std(hv_values, ddof=1) if len(hv_values) > 1 else 0.0,
                            'gd_mean': np.mean(gd_values), 'gd_std': np.std(gd_values, ddof=1) if len(gd_values) > 1 else 0.0,
                            'composite_mean': np.mean(composite_values), 'composite_std': np.std(composite_values, ddof=1) if len(composite_values) > 1 else 0.0,
                            'n_valid_runs': len(igd_values),
                        }
                    else:
                        results[case_no][alg_name] = {'igd_mean': float('inf'), 'hv_mean': 0.0, 'gd_mean': float('inf'), 'composite_mean': float('inf'), 'composite_std': 0.0, 'n_valid_runs': 0}
            
            self.log.emit("\n=== 多进程加速试验全部完成 ===")
            self.finished_result.emit(results)
            
        except Exception as e:
            import traceback
            self.error.emit(f"并行执行错误: {str(e)}\n{traceback.format_exc()}")

