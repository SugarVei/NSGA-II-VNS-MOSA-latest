# -*- coding: utf-8 -*-
"""
田口实验 CLI 入口
Taguchi Experiment CLI Entry Point

运行田口实验并生成所有结果文件。
"""

import sys
import os
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from algorithms.nsga2 import NSGAII
from algorithms.mosa import MOSA

from .designs import (
    FACTORS, L16_ARRAY, REAL_CASE_PARAMS, FIXED_HYBRID_PARAMS,
    get_params_for_run, get_level_indices_for_run, get_n_runs
)
from .pareto import build_pf_ref, get_non_dominated
from .metrics import (
    normalize_objectives, compute_igd, compute_gd, compute_hv,
    get_hv_ref_point, get_normalization_info, compute_all_metrics
)
from .analysis import (
    compute_all_snr_effects, compute_summary_statistics,
    recommend_best_params, get_default_params, generate_paper_summary
)
from .plotting import create_all_figures
from .io import (
    save_csv, save_json, save_pf_ref, save_normalization, save_hv_ref_point,
    create_output_structure, ErrorLogger
)


def set_all_seeds(seed: int) -> None:
    """
    设置所有随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)


def create_problem_instance() -> SchedulingProblem:
    """
    创建固定的问题实例（用于田口实验）
    
    Returns:
        调度问题实例
    """
    return SchedulingProblem.generate_random(
        n_jobs=REAL_CASE_PARAMS['n_jobs'],
        n_stages=REAL_CASE_PARAMS['n_stages'],
        machines_per_stage=REAL_CASE_PARAMS['machines_per_stage'],
        n_speed_levels=REAL_CASE_PARAMS['n_speed_levels'],
        n_skill_levels=REAL_CASE_PARAMS['n_skill_levels'],
        seed=42  # 固定种子确保问题实例一致
    )


def run_hybrid(problem: SchedulingProblem, 
               params: Dict[str, Any], 
               seed: int) -> Tuple[np.ndarray, float]:
    """
    运行 HYBRID 算法 (NSGA-II + MOSA)
    
    Args:
        problem: 调度问题实例
        params: 算法参数
        seed: 随机种子
        
    Returns:
        (objectives, time_sec): 非支配解目标点数组和运行时间
    """
    set_all_seeds(seed)
    
    start_time = time.time()
    
    # 运行 NSGA-II
    nsga2 = NSGAII(
        problem=problem,
        pop_size=params['population_size'],
        n_generations=params.get('n_generations', FIXED_HYBRID_PARAMS['n_generations']),
        crossover_prob=params['crossover_prob'],
        mutation_prob=params['mutation_prob'],
        seed=seed
    )
    pareto_nsga = nsga2.run()
    
    # 运行 MOSA
    mosa = MOSA(
        problem=problem,
        initial_temp=params['initial_temp'],
        cooling_rate=params.get('cooling_rate', FIXED_HYBRID_PARAMS['cooling_rate']),
        final_temp=params.get('final_temp', FIXED_HYBRID_PARAMS['final_temp']),
        max_iterations=params.get('max_iterations', FIXED_HYBRID_PARAMS['max_iterations']),
        vns_iterations=params.get('vns_iterations', FIXED_HYBRID_PARAMS['vns_iterations']),
        n_representative=params.get('n_representative', FIXED_HYBRID_PARAMS['n_representative']),
        seed=seed
    )
    final_archive = mosa.run(pareto_nsga)
    
    elapsed_time = time.time() - start_time
    
    # 提取目标值
    objectives = np.array([s.objectives for s in final_archive if s.objectives is not None])
    
    return objectives, elapsed_time


def run_taguchi_experiment(problem: SchedulingProblem,
                           n_rep: int = 30,
                           base_seed: int = 20250101,
                           n_jobs: int = 1,
                           error_logger: Optional[ErrorLogger] = None,
                           verbose: bool = True) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    运行完整的田口实验
    
    Args:
        problem: 调度问题实例
        n_rep: 每组重复次数
        base_seed: 基础随机种子
        n_jobs: 并行任务数 (暂不实现并行)
        error_logger: 异常记录器
        verbose: 是否显示进度
        
    Returns:
        (runs_df, all_objectives): 运行结果 DataFrame 和所有目标点列表
    """
    n_runs = get_n_runs()
    total_runs = n_runs * n_rep
    
    results = []
    all_objectives = []
    
    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc="Taguchi Runs", total=n_runs)
    
    for run_id in iterator:
        params = get_params_for_run(run_id)
        level_indices = get_level_indices_for_run(run_id)
        
        for rep_id in range(n_rep):
            seed = base_seed + run_id * 1000 + rep_id
            
            try:
                objectives, time_sec = run_hybrid(problem, params, seed)
                n_solutions = len(objectives)
                
                # 保存目标点用于构造 PF_ref
                if len(objectives) > 0:
                    all_objectives.append(objectives)
                
                result = {
                    'run_id': run_id,
                    'rep_id': rep_id,
                    **level_indices,
                    'population_size': params['population_size'],
                    'crossover_prob': params['crossover_prob'],
                    'mutation_prob': params['mutation_prob'],
                    'initial_temp': params['initial_temp'],
                    'seed': seed,
                    'time_sec': time_sec,
                    'n_solutions': n_solutions,
                    # 指标稍后计算
                    'igd': np.nan,
                    'gd': np.nan,
                    'hv': np.nan,
                }
                
            except Exception as e:
                if error_logger:
                    error_logger.log_error(run_id, rep_id, params, seed, e)
                
                result = {
                    'run_id': run_id,
                    'rep_id': rep_id,
                    **level_indices,
                    'population_size': params['population_size'],
                    'crossover_prob': params['crossover_prob'],
                    'mutation_prob': params['mutation_prob'],
                    'initial_temp': params['initial_temp'],
                    'seed': seed,
                    'time_sec': np.nan,
                    'n_solutions': 0,
                    'igd': np.nan,
                    'gd': np.nan,
                    'hv': np.nan,
                }
                all_objectives.append(np.array([]).reshape(0, 3))
            
            results.append(result)
    
    return pd.DataFrame(results), all_objectives


def compute_metrics_for_runs(df: pd.DataFrame,
                             all_objectives: List[np.ndarray],
                             pf_ref: np.ndarray,
                             f_min: np.ndarray,
                             f_max: np.ndarray,
                             hv_ref_point: np.ndarray) -> pd.DataFrame:
    """
    为所有运行计算指标
    
    Args:
        df: 运行结果 DataFrame
        all_objectives: 所有运行的目标点列表
        pf_ref: 参考前沿
        f_min: 归一化最小值
        f_max: 归一化最大值
        hv_ref_point: HV 参考点
        
    Returns:
        更新后的 DataFrame
    """
    df = df.copy()
    
    pf_ref_norm = normalize_objectives(pf_ref, f_min, f_max)
    
    for idx, objectives in enumerate(all_objectives):
        if len(objectives) == 0:
            continue
        
        metrics = compute_all_metrics(objectives, pf_ref, f_min, f_max, hv_ref_point)
        
        df.loc[idx, 'igd'] = metrics['igd']
        df.loc[idx, 'gd'] = metrics['gd']
        df.loc[idx, 'hv'] = metrics['hv']
    
    return df


def run_confirmation_experiment(problem: SchedulingProblem,
                                best_params: Dict[str, Any],
                                n_rep: int,
                                base_seed: int,
                                pf_ref: np.ndarray,
                                f_min: np.ndarray,
                                f_max: np.ndarray,
                                hv_ref_point: np.ndarray,
                                error_logger: Optional[ErrorLogger] = None,
                                verbose: bool = True) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    运行确认实验
    
    Args:
        problem: 调度问题实例
        best_params: 最优参数
        n_rep: 重复次数
        base_seed: 基础种子
        pf_ref: 参考前沿
        f_min: 归一化最小值
        f_max: 归一化最大值
        hv_ref_point: HV 参考点
        error_logger: 异常记录器
        verbose: 是否显示进度
        
    Returns:
        (confirm_df, all_objectives): 确认实验结果和目标点列表
    """
    results = []
    all_objectives = []
    
    iterator = range(n_rep)
    if verbose:
        iterator = tqdm(iterator, desc="Confirmation Experiment", total=n_rep)
    
    for rep_id in iterator:
        seed = base_seed + 100000 + rep_id
        
        try:
            objectives, time_sec = run_hybrid(problem, best_params, seed)
            n_solutions = len(objectives)
            
            if len(objectives) > 0:
                all_objectives.append(objectives)
                metrics = compute_all_metrics(objectives, pf_ref, f_min, f_max, hv_ref_point)
            else:
                metrics = {'igd': np.nan, 'gd': np.nan, 'hv': np.nan}
            
            result = {
                'rep_id': rep_id,
                'seed': seed,
                'time_sec': time_sec,
                'n_solutions': n_solutions,
                **metrics,
            }
            
        except Exception as e:
            if error_logger:
                error_logger.log_error(-1, rep_id, best_params, seed, e)
            
            result = {
                'rep_id': rep_id,
                'seed': seed,
                'time_sec': np.nan,
                'n_solutions': 0,
                'igd': np.nan,
                'gd': np.nan,
                'hv': np.nan,
            }
            all_objectives.append(np.array([]).reshape(0, 3))
        
        results.append(result)
    
    return pd.DataFrame(results), all_objectives


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='田口实验 - HYBRID 算法参数调优',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--case', type=str, default='real_case_15x3',
                        help='问题实例名称 (默认: real_case_15x3)')
    parser.add_argument('--rep', type=int, default=30,
                        help='每组重复次数 (默认: 30)')
    parser.add_argument('--base-seed', type=int, default=20250101,
                        help='基础随机种子 (默认: 20250101)')
    parser.add_argument('--out', type=str, default='results/taguchi',
                        help='输出目录 (默认: results/taguchi)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='并行任务数 (默认: 1)')
    parser.add_argument('--skip-confirm', action='store_true',
                        help='跳过确认实验')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("田口实验 - HYBRID 算法参数调优")
        print("=" * 60)
        print(f"问题实例: {args.case}")
        print(f"重复次数: {args.rep}")
        print(f"基础种子: {args.base_seed}")
        print(f"输出目录: {args.out}")
        print("=" * 60)
    
    # 创建输出目录结构
    paths = create_output_structure(Path(args.out))
    
    # 创建异常记录器
    error_logger = ErrorLogger(paths['error_log'])
    
    # 创建问题实例
    if verbose:
        print("\n[1/7] 创建问题实例...")
    problem = create_problem_instance()
    
    # 设置工人可用数量
    problem.workers_available = REAL_CASE_PARAMS['workers_available']
    
    if verbose:
        print(f"  - 工件数: {problem.n_jobs}")
        print(f"  - 阶段数: {problem.n_stages}")
        print(f"  - 每阶段机器: {problem.machines_per_stage}")
        print(f"  - 工人技能人数: {problem.workers_available}")
    
    # 运行田口实验
    if verbose:
        print(f"\n[2/7] 运行田口实验 (16 × {args.rep} = {16 * args.rep} 次)...")
    
    runs_df, all_objectives = run_taguchi_experiment(
        problem=problem,
        n_rep=args.rep,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
        error_logger=error_logger,
        verbose=verbose
    )
    
    # 构造 PF_ref
    if verbose:
        print("\n[3/7] 构造全局参考前沿 PF_ref...")
    
    pf_ref = build_pf_ref(all_objectives)
    
    if verbose:
        print(f"  - 合并 {sum(len(o) for o in all_objectives)} 个目标点")
        print(f"  - PF_ref 包含 {len(pf_ref)} 个非支配点")
    
    # 保存 PF_ref
    save_pf_ref(pf_ref, paths['pf_ref'])
    
    # 计算归一化参数
    norm_info = get_normalization_info(pf_ref)
    f_min = np.array(norm_info['f_min'])
    f_max = np.array(norm_info['f_max'])
    hv_ref_point = np.array(norm_info['hv_ref_point'])
    
    save_normalization(f_min, f_max, paths['normalization'])
    save_hv_ref_point(hv_ref_point, paths['hv_ref_point'])
    
    if verbose:
        print(f"  - f_min: {f_min}")
        print(f"  - f_max: {f_max}")
        print(f"  - HV ref_point: {hv_ref_point}")
    
    # 计算所有运行的指标
    if verbose:
        print("\n[4/7] 计算 IGD/GD/HV 指标...")
    
    runs_df = compute_metrics_for_runs(
        runs_df, all_objectives, pf_ref, f_min, f_max, hv_ref_point
    )
    
    # 保存运行结果
    save_csv(runs_df, paths['taguchi_runs'])
    
    # 计算汇总统计（使用样本标准差 ddof=1）
    if verbose:
        print("\n计算汇总统计...")
    summary_df = compute_summary_statistics(runs_df, n_rep=args.rep, verbose=verbose)
    save_csv(summary_df, paths['taguchi_summary'])
    
    # 生成论文专用格式表
    paper_summary_df = generate_paper_summary(summary_df)
    save_csv(paper_summary_df, paths['taguchi_summary_paper'])
    
    if verbose:
        print(f"  - 生成 taguchi_summary.csv（结构化数值表）")
        print(f"  - 生成 taguchi_summary_paper.csv（论文 Mean(std.) 格式）")
    
    # 计算主效应
    if verbose:
        print("\n[5/7] 田口分析 (SNR 主效应)...")
    
    effects = compute_all_snr_effects(runs_df)
    
    # 保存主效应表
    save_csv(effects['igd'], paths['snr_effect_igd'], index=True)
    save_csv(effects['hv'], paths['snr_effect_hv'], index=True)
    save_csv(effects['gd'], paths['snr_effect_gd'], index=True)
    
    # 推荐最优参数
    best_result = recommend_best_params(effects['igd'], effects['hv'], effects['gd'])
    save_json(best_result, paths['best_params'])
    
    if verbose:
        print(f"  - 推荐最优水平: {best_result['best_levels']}")
        print(f"  - 推荐最优参数: {best_result['best_params']}")
    
    # 确认实验
    confirm_df = None
    default_df = None
    best_run_objectives = np.array([]).reshape(0, 3)
    
    if not args.skip_confirm:
        if verbose:
            print(f"\n[6/7] 确认实验 ({args.rep} 次)...")
        
        # 运行最优参数
        confirm_df, confirm_objectives = run_confirmation_experiment(
            problem=problem,
            best_params=best_result['best_params'],
            n_rep=args.rep,
            base_seed=args.base_seed,
            pf_ref=pf_ref,
            f_min=f_min,
            f_max=f_max,
            hv_ref_point=hv_ref_point,
            error_logger=error_logger,
            verbose=verbose
        )
        
        if len(confirm_objectives) > 0:
            best_run_objectives = confirm_objectives[len(confirm_objectives) // 2]
        
        save_csv(confirm_df, paths['confirm_summary'])
        
        # 运行默认参数对比
        default_params = get_default_params()
        default_df, _ = run_confirmation_experiment(
            problem=problem,
            best_params=default_params,
            n_rep=args.rep,
            base_seed=args.base_seed,
            pf_ref=pf_ref,
            f_min=f_min,
            f_max=f_max,
            hv_ref_point=hv_ref_point,
            error_logger=error_logger,
            verbose=verbose
        )
        
        if verbose:
            print(f"  最优参数 IGD: {confirm_df['igd'].mean():.6f} ± {confirm_df['igd'].std():.6f}")
            print(f"  默认参数 IGD: {default_df['igd'].mean():.6f} ± {default_df['igd'].std():.6f}")
    else:
        if verbose:
            print("\n[6/7] 跳过确认实验")
        
        # 取第一个运行的目标点用于绘图
        for obj in all_objectives:
            if len(obj) > 0:
                best_run_objectives = obj
                break
    
    # 生成图表
    if verbose:
        print("\n[7/7] 生成图表...")
    
    created_figures = create_all_figures(
        df=runs_df,
        effects=effects,
        pf_ref=pf_ref,
        best_run_objectives=best_run_objectives,
        figures_dir=paths['figures'],
        confirm_df=confirm_df,
        default_df=default_df
    )
    
    if verbose:
        print(f"  - 生成 {len(created_figures)} 张图表")
    
    # 完成日志
    error_logger.finalize()
    
    if verbose:
        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)
        print(f"输出目录: {paths['base']}")
        print(f"错误数量: {error_logger.get_error_count()}")
        print("\n生成的文件:")
        for key, path in paths.items():
            if key not in ['base', 'figures'] and Path(path).exists():
                print(f"  - {path}")
        print(f"\n图表目录: {paths['figures']}")
        for fig_path in created_figures:
            print(f"  - {fig_path.name}")


if __name__ == '__main__':
    main()
