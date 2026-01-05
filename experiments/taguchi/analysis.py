# -*- coding: utf-8 -*-
"""
田口分析模块
Taguchi Analysis Module

SNR 计算、主效应分析、最优组合推荐和确认实验。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any

from .designs import FACTORS, L16_ARRAY, get_factor_names, get_n_levels


def compute_snr_smaller_better(y: np.ndarray) -> float:
    """
    Smaller-the-better S/N 比计算
    
    SNR = -10 * log10( mean( y^2 ) )
    
    适用于 IGD、GD 等越小越好的指标。
    
    Args:
        y: 指标值数组
        
    Returns:
        S/N 比值
    """
    y = np.asarray(y)
    y = y[~np.isnan(y)]  # 移除 NaN
    
    if len(y) == 0:
        return float('-inf')
    
    # 避免 log(0)
    mean_sq = np.mean(y ** 2)
    if mean_sq < 1e-12:
        mean_sq = 1e-12
    
    return -10 * np.log10(mean_sq)


def compute_snr_larger_better(y: np.ndarray) -> float:
    """
    Larger-the-better S/N 比计算
    
    SNR = -10 * log10( mean( 1/y^2 ) )
    
    适用于 HV 等越大越好的指标。
    
    Args:
        y: 指标值数组
        
    Returns:
        S/N 比值
    """
    y = np.asarray(y)
    y = y[~np.isnan(y)]  # 移除 NaN
    y = y[y > 1e-12]     # 移除零和负值
    
    if len(y) == 0:
        return float('-inf')
    
    mean_inv_sq = np.mean(1.0 / (y ** 2))
    if mean_inv_sq < 1e-12:
        mean_inv_sq = 1e-12
    
    return -10 * np.log10(mean_inv_sq)


def compute_main_effects(df: pd.DataFrame, 
                         metric: str, 
                         snr_func: Callable[[np.ndarray], float]) -> pd.DataFrame:
    """
    计算主效应表
    
    对每个因子的每个水平，计算使用该水平的所有运行的 SNR 均值。
    
    Args:
        df: 包含运行结果的 DataFrame，必须有列：
            A_level, B_level, C_level, D_level, 以及指标列
        metric: 指标列名 (如 'igd', 'hv', 'gd')
        snr_func: S/N 比计算函数
        
    Returns:
        主效应表 DataFrame，行为因子，列为水平
    """
    factors = get_factor_names()
    n_levels = get_n_levels()
    
    effects = {}
    
    for factor in factors:
        level_col = f'{factor}_level'
        factor_effects = []
        
        for level in range(1, n_levels + 1):
            # 找出使用该水平的所有运行
            mask = df[level_col] == level
            values = df.loc[mask, metric].values
            
            if len(values) > 0:
                snr = snr_func(values)
            else:
                snr = float('nan')
            
            factor_effects.append(snr)
        
        effects[factor] = factor_effects
    
    # 创建 DataFrame
    result = pd.DataFrame(effects, index=[f'Level {i}' for i in range(1, n_levels + 1)])
    result = result.T
    result.index.name = 'Factor'
    
    return result


def recommend_best_params(effects_igd: pd.DataFrame,
                          effects_hv: pd.DataFrame,
                          effects_gd: pd.DataFrame,
                          weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> Dict[str, Any]:
    """
    推荐最优参数组合
    
    默认策略：
    1. 首先最大化 IGD 的 SNR（越大越好）
    2. 其次最大化 HV 的 SNR（越大越好）
    3. 最后最大化 GD 的 SNR（越大越好）
    
    支持加权综合评分。
    
    Args:
        effects_igd: IGD 主效应表
        effects_hv: HV 主效应表
        effects_gd: GD 主效应表
        weights: (w_igd, w_hv, w_gd) 权重
        
    Returns:
        包含推荐参数的字典
    """
    factors = get_factor_names()
    n_levels = get_n_levels()
    
    best_levels = {}
    detailed_scores = {}
    
    for factor in factors:
        # 归一化各指标的 SNR
        igd_vals = effects_igd.loc[factor].values
        hv_vals = effects_hv.loc[factor].values
        gd_vals = effects_gd.loc[factor].values
        
        # 归一化到 [0, 1]
        def normalize(vals):
            vals = np.array(vals)
            if np.all(np.isnan(vals)):
                return np.zeros_like(vals)
            min_v, max_v = np.nanmin(vals), np.nanmax(vals)
            if max_v - min_v < 1e-12:
                return np.ones_like(vals)
            return (vals - min_v) / (max_v - min_v + 1e-12)
        
        igd_norm = normalize(igd_vals)
        hv_norm = normalize(hv_vals)
        gd_norm = normalize(gd_vals)
        
        # 加权综合得分
        scores = weights[0] * igd_norm + weights[1] * hv_norm + weights[2] * gd_norm
        
        best_level = int(np.nanargmax(scores) + 1)
        best_levels[factor] = best_level
        detailed_scores[factor] = {
            'scores': scores.tolist(),
            'best_level': best_level,
            'igd_snr': igd_vals.tolist(),
            'hv_snr': hv_vals.tolist(),
            'gd_snr': gd_vals.tolist(),
        }
    
    # 转换为实际参数值
    best_params = {}
    for factor, level in best_levels.items():
        param_name = FACTORS[factor]['name']
        param_value = FACTORS[factor]['levels'][level - 1]
        best_params[param_name] = param_value
    
    return {
        'best_levels': best_levels,
        'best_params': best_params,
        'detailed_scores': detailed_scores,
        'weights': {'igd': weights[0], 'hv': weights[1], 'gd': weights[2]},
    }


def get_default_params() -> Dict[str, Any]:
    """
    获取默认参数组合（用于对比）
    
    Returns:
        默认参数字典
    """
    return {
        'population_size': 100,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1,
        'initial_temp': 100,
    }


def compute_summary_statistics(df: pd.DataFrame, 
                               metrics: List[str] = ['igd', 'gd', 'hv'],
                               n_rep: int = 30,
                               warn_threshold: float = 0.8,
                               verbose: bool = True) -> pd.DataFrame:
    """
    计算按 run_id 分组的汇总统计
    
    使用 pandas groupby 聚合，使用样本标准差 (ddof=1) 更符合论文习惯。
    
    Args:
        df: 运行结果 DataFrame
        metrics: 要统计的指标列表
        n_rep: 预期的重复次数（用于计算有效重复率）
        warn_threshold: 有效样本比例警告阈值（默认 0.8 即 80%）
        verbose: 是否打印警告信息
        
    Returns:
        汇总 DataFrame，按 run_id 排序
    """
    # 定义聚合函数：mean 和 std（使用默认 ddof=1 的样本标准差）
    agg_dict = {}
    
    # 因子水平和参数列：取第一个值即可（同一 run_id 这些值都相同）
    first_cols = ['A_level', 'B_level', 'C_level', 'D_level',
                  'population_size', 'crossover_prob', 'mutation_prob', 'initial_temp']
    for col in first_cols:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    # 指标列：计算 mean 和 std
    for metric in metrics:
        if metric in df.columns:
            agg_dict[metric] = ['mean', 'std', 'count']
    
    # 其他指标：time_sec, n_solutions
    if 'time_sec' in df.columns:
        agg_dict['time_sec'] = ['mean', 'std']
    if 'n_solutions' in df.columns:
        agg_dict['n_solutions'] = ['mean', 'std']
    
    # 执行 groupby 聚合
    grouped = df.groupby('run_id').agg(agg_dict)
    
    # 扁平化列名
    result_data = []
    for run_id in sorted(grouped.index):
        row_data = grouped.loc[run_id]
        row = {'run_id': run_id}
        
        # 添加因子水平和参数值
        for col in first_cols:
            if col in df.columns:
                row[col] = row_data[(col, 'first')]
        
        # 添加指标的 mean, std
        n_valid = None
        for metric in metrics:
            if metric in df.columns:
                row[f'{metric}_mean'] = row_data[(metric, 'mean')]
                row[f'{metric}_std'] = row_data[(metric, 'std')]
                # 使用第一个指标的 count 作为 n_valid
                if n_valid is None:
                    n_valid = row_data[(metric, 'count')]
        
        # 添加时间和解数量
        if 'time_sec' in df.columns:
            row['time_mean'] = row_data[('time_sec', 'mean')]
            row['time_std'] = row_data[('time_sec', 'std')]
        if 'n_solutions' in df.columns:
            row['n_solutions_mean'] = row_data[('n_solutions', 'mean')]
            row['n_solutions_std'] = row_data[('n_solutions', 'std')]
        
        # 添加有效重复次数
        row['n_valid'] = int(n_valid) if n_valid is not None else 0
        
        # 检查有效样本是否不足
        if verbose and n_valid is not None:
            valid_ratio = n_valid / n_rep
            if valid_ratio < warn_threshold:
                print(f"[WARN] run_id={run_id} valid reps = {int(n_valid)} / {n_rep}")
        
        result_data.append(row)
    
    # 构建 DataFrame 并按 run_id 排序
    summary_df = pd.DataFrame(result_data)
    
    # 确保列顺序
    col_order = ['run_id', 'A_level', 'B_level', 'C_level', 'D_level',
                 'population_size', 'crossover_prob', 'mutation_prob', 'initial_temp']
    for metric in metrics:
        col_order.extend([f'{metric}_mean', f'{metric}_std'])
    col_order.extend(['time_mean', 'time_std', 'n_solutions_mean', 'n_solutions_std', 'n_valid'])
    
    # 只保留存在的列
    col_order = [c for c in col_order if c in summary_df.columns]
    summary_df = summary_df[col_order]
    
    return summary_df.sort_values('run_id').reset_index(drop=True)


def generate_paper_summary(summary_df: pd.DataFrame,
                           metrics: List[str] = ['igd', 'gd', 'hv'],
                           precision: int = 6) -> pd.DataFrame:
    """
    生成论文专用格式的汇总表
    
    将 Mean 和 Std 合并为 "Mean (std.)" 格式，方便论文直接使用。
    
    Args:
        summary_df: 汇总统计 DataFrame（由 compute_summary_statistics 生成）
        metrics: 要格式化的指标列表
        precision: 小数位数（默认 6 位）
        
    Returns:
        论文格式的汇总 DataFrame
    """
    paper_data = []
    
    for _, row in summary_df.iterrows():
        paper_row = {
            'run_id': int(row['run_id']),
        }
        
        # 添加因子水平（如果存在）
        for col in ['A_level', 'B_level', 'C_level', 'D_level']:
            if col in row:
                paper_row[col] = int(row[col])
        
        # 添加参数值（如果存在）
        for col in ['population_size', 'crossover_prob', 'mutation_prob', 'initial_temp']:
            if col in row:
                paper_row[col] = row[col]
        
        # 格式化指标为 "Mean (std.)"
        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in row and std_col in row:
                mean_val = row[mean_col]
                std_val = row[std_col]
                
                # 处理 NaN
                if pd.isna(mean_val) or pd.isna(std_val):
                    paper_row[f'{metric.upper()} Mean (std.)'] = 'N/A'
                else:
                    paper_row[f'{metric.upper()} Mean (std.)'] = f"{mean_val:.{precision}f} ({std_val:.{precision}f})"
        
        # 添加时间（如果存在）
        if 'time_mean' in row and 'time_std' in row:
            if pd.isna(row['time_mean']) or pd.isna(row['time_std']):
                paper_row['Time Mean (std.)'] = 'N/A'
            else:
                paper_row['Time Mean (std.)'] = f"{row['time_mean']:.2f} ({row['time_std']:.2f})"
        
        # 添加解数量（如果存在）
        if 'n_solutions_mean' in row and 'n_solutions_std' in row:
            if pd.isna(row['n_solutions_mean']) or pd.isna(row['n_solutions_std']):
                paper_row['Solutions Mean (std.)'] = 'N/A'
            else:
                paper_row['Solutions Mean (std.)'] = f"{row['n_solutions_mean']:.1f} ({row['n_solutions_std']:.1f})"
        
        # 添加有效重复次数
        if 'n_valid' in row:
            paper_row['n_valid'] = int(row['n_valid'])
        
        paper_data.append(paper_row)
    
    return pd.DataFrame(paper_data)


def compute_all_snr_effects(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    计算所有指标的主效应表
    
    Args:
        df: 运行结果 DataFrame
        
    Returns:
        {'igd': df_igd, 'hv': df_hv, 'gd': df_gd}
    """
    return {
        'igd': compute_main_effects(df, 'igd', compute_snr_smaller_better),
        'hv': compute_main_effects(df, 'hv', compute_snr_larger_better),
        'gd': compute_main_effects(df, 'gd', compute_snr_smaller_better),
    }
