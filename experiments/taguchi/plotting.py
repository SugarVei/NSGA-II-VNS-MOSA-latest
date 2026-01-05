# -*- coding: utf-8 -*-
"""
绘图模块
Plotting Module

田口实验结果可视化，包括主效应图、箱线图和 Pareto 投影图。
"""

import numpy as np
import pandas as pd

# 设置 matplotlib 后端（必须在导入 pyplot 之前）
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于文件保存

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from typing import List, Optional, Tuple

from .designs import FACTORS, get_factor_names, get_n_levels

# 设置中文字体支持
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def setup_plot_style():
    """设置论文友好的绘图样式"""
    # 使用兼容的样式设置
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')  # fallback
    
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['figure.figsize'] = (10, 6)
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['legend.fontsize'] = 11


def plot_main_effects(effects_df: pd.DataFrame, 
                      metric_name: str, 
                      save_path: Path,
                      title_suffix: str = '') -> None:
    """
    绘制主效应图
    
    X轴为因子水平，Y轴为 SNR 均值，每个因子一条折线。
    
    Args:
        effects_df: 主效应表，行为因子，列为水平
        metric_name: 指标名称 (用于标题)
        save_path: 保存路径
        title_suffix: 标题后缀
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factors = effects_df.index.tolist()
    n_levels = len(effects_df.columns)
    levels = list(range(1, n_levels + 1))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, factor in enumerate(factors):
        values = effects_df.loc[factor].values
        param_name = FACTORS[factor]['name']
        level_values = FACTORS[factor]['levels']
        
        # 创建 x 标签
        x_labels = [f'{level_values[j]}' for j in range(n_levels)]
        
        ax.plot(levels, values, 
                marker=markers[i], 
                color=colors[i],
                linewidth=2,
                markersize=8,
                label=f'{factor}: {param_name}')
    
    ax.set_xlabel('Factor Level', fontsize=14)
    ax.set_ylabel('S/N Ratio (dB)', fontsize=14)
    ax.set_title(f'Main Effects Plot for {metric_name.upper()}{title_suffix}', fontsize=16)
    ax.set_xticks(levels)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_boxplot_by_run(df: pd.DataFrame, 
                        metric: str, 
                        save_path: Path,
                        title_suffix: str = '') -> None:
    """
    绘制按运行分组的箱线图
    
    X轴为 Run1-Run16，Y轴为指标值。
    
    Args:
        df: 运行结果 DataFrame，必须有 run_id 和指标列
        metric: 指标列名
        save_path: 保存路径
        title_suffix: 标题后缀
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 准备数据
    run_data = []
    for run_id in range(16):
        values = df[df['run_id'] == run_id][metric].values
        run_data.append(values)
    
    # 绘制箱线图
    bp = ax.boxplot(run_data, 
                    positions=range(1, 17),
                    widths=0.6,
                    patch_artist=True)
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 16))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Run ID', fontsize=14)
    ax.set_ylabel(f'{metric.upper()} Value', fontsize=14)
    ax.set_title(f'Distribution of {metric.upper()} across Runs{title_suffix}', fontsize=16)
    ax.set_xticks(range(1, 17))
    ax.set_xticklabels([f'R{i}' for i in range(1, 17)])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pareto_projections(A: np.ndarray, 
                            pf_ref: np.ndarray, 
                            save_dir: Path,
                            prefix: str = 'pareto') -> None:
    """
    绘制 Pareto 前沿的三个二维投影
    
    (F1, F2), (F1, F3), (F2, F3)
    
    Args:
        A: 解集目标点 (n, 3)
        pf_ref: 参考前沿 (m, 3)
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    setup_plot_style()
    
    projections = [
        (0, 1, 'F1 (Makespan)', 'F2 (Labor Cost)', 'f1_f2'),
        (0, 2, 'F1 (Makespan)', 'F3 (Energy)', 'f1_f3'),
        (1, 2, 'F2 (Labor Cost)', 'F3 (Energy)', 'f2_f3'),
    ]
    
    for i, j, xlabel, ylabel, suffix in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制 PF_ref
        if len(pf_ref) > 0:
            ax.scatter(pf_ref[:, i], pf_ref[:, j], 
                      c='#2ca02c', marker='o', s=50, alpha=0.6,
                      label=f'PF_ref (n={len(pf_ref)})', zorder=1)
        
        # 绘制解集 A
        if len(A) > 0:
            ax.scatter(A[:, i], A[:, j], 
                      c='#d62728', marker='x', s=80, alpha=0.8,
                      label=f'Solution Set (n={len(A)})', zorder=2)
        
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'Pareto Front Projection: {xlabel} vs {ylabel}', fontsize=14)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / f'{prefix}_{suffix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_pareto_3d(A: np.ndarray, 
                   pf_ref: np.ndarray, 
                   save_path: Path) -> None:
    """
    绘制 3D Pareto 前沿图
    
    Args:
        A: 解集目标点 (n, 3)
        pf_ref: 参考前沿 (m, 3)
        save_path: 保存路径
    """
    setup_plot_style()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制 PF_ref
    if len(pf_ref) > 0:
        ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], 
                  c='#2ca02c', marker='o', s=30, alpha=0.5,
                  label=f'PF_ref (n={len(pf_ref)})')
    
    # 绘制解集 A
    if len(A) > 0:
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], 
                  c='#d62728', marker='x', s=60, alpha=0.8,
                  label=f'Solution Set (n={len(A)})')
    
    ax.set_xlabel('F1 (Makespan)', fontsize=12)
    ax.set_ylabel('F2 (Labor Cost)', fontsize=12)
    ax.set_zlabel('F3 (Energy)', fontsize=12)
    ax.set_title('3D Pareto Front Visualization', fontsize=14)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confirmation_comparison(confirm_df: pd.DataFrame,
                                 default_df: Optional[pd.DataFrame],
                                 save_path: Path) -> None:
    """
    绘制确认实验与默认参数的对比图
    
    Args:
        confirm_df: 确认实验结果
        default_df: 默认参数结果 (可选)
        save_path: 保存路径
    """
    setup_plot_style()
    
    metrics = ['igd', 'gd', 'hv']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, metrics):
        data = [confirm_df[metric].values]
        labels = ['Optimal']
        
        if default_df is not None:
            data.append(default_df[metric].values)
            labels.append('Default')
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        colors = ['#2ca02c', '#1f77b4']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Confirmation Experiment: Optimal vs Default Parameters', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_figures(df: pd.DataFrame,
                       effects: dict,
                       pf_ref: np.ndarray,
                       best_run_objectives: np.ndarray,
                       figures_dir: Path,
                       confirm_df: Optional[pd.DataFrame] = None,
                       default_df: Optional[pd.DataFrame] = None) -> List[Path]:
    """
    创建所有必需的图表
    
    Args:
        df: 运行结果 DataFrame
        effects: {'igd': df, 'hv': df, 'gd': df} 主效应表字典
        pf_ref: 参考前沿
        best_run_objectives: 最优参数运行的目标点
        figures_dir: 图表保存目录
        confirm_df: 确认实验结果
        default_df: 默认参数结果
        
    Returns:
        生成的图表文件路径列表
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # 1. 主效应图
    for metric in ['igd', 'hv', 'gd']:
        path = figures_dir / f'main_effect_{metric}.png'
        plot_main_effects(effects[metric], metric, path)
        created_files.append(path)
    
    # 2. 箱线图
    for metric in ['igd', 'hv', 'gd']:
        path = figures_dir / f'boxplot_{metric}.png'
        plot_boxplot_by_run(df, metric, path)
        created_files.append(path)
    
    # 3. Pareto 投影图
    plot_pareto_projections(best_run_objectives, pf_ref, figures_dir)
    for suffix in ['f1_f2', 'f1_f3', 'f2_f3']:
        created_files.append(figures_dir / f'pareto_{suffix}.png')
    
    # 4. 3D Pareto 图 (可选)
    path_3d = figures_dir / 'pareto_3d.png'
    plot_pareto_3d(best_run_objectives, pf_ref, path_3d)
    created_files.append(path_3d)
    
    # 5. 确认实验对比图
    if confirm_df is not None:
        path = figures_dir / 'confirmation_comparison.png'
        plot_confirmation_comparison(confirm_df, default_df, path)
        created_files.append(path)
    
    return created_files
