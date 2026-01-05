"""
收敛曲线可视化模块
Convergence Curve Visualization

绘制算法收敛过程中三个目标函数的变化曲线。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Optional, Tuple
import os


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_convergence(convergence_data: Dict,
                     algorithm_name: str = "Algorithm",
                     figsize: Tuple[int, int] = (12, 8),
                     save_path: Optional[str] = None) -> Figure:
    """
    绘制收敛曲线
    
    Args:
        convergence_data: 收敛历史数据字典，包含:
            - 'generation' 或 'iteration': 迭代次数
            - 'best_makespan': 最优Makespan
            - 'best_labor_cost': 最优人工成本
            - 'best_energy': 最优能耗
        algorithm_name: 算法名称
        figsize: 图像大小
        save_path: 保存路径 (可选)
        
    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{algorithm_name} 收敛曲线', fontsize=14, fontweight='bold')
    
    # 获取迭代次数
    iterations = convergence_data.get('generation') or convergence_data.get('iteration', [])
    
    if not iterations:
        return fig
    
    # 颜色方案
    colors = {
        'makespan': '#2196F3',      # 蓝色
        'labor_cost': '#4CAF50',    # 绿色
        'energy': '#FF9800',        # 橙色
        'pareto': '#9C27B0'         # 紫色
    }
    
    # 1. Makespan收敛曲线
    ax1 = axes[0, 0]
    makespan = convergence_data.get('best_makespan', [])
    if makespan:
        ax1.plot(iterations, makespan, color=colors['makespan'], linewidth=2, label='最优值')
        if 'avg_makespan' in convergence_data:
            ax1.plot(iterations, convergence_data['avg_makespan'], 
                    color=colors['makespan'], linewidth=1, linestyle='--', 
                    alpha=0.6, label='平均值')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('Makespan (分钟)')
    ax1.set_title('最大完工时间 (F1)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Labor Cost收敛曲线
    ax2 = axes[0, 1]
    labor_cost = convergence_data.get('best_labor_cost', [])
    if labor_cost:
        ax2.plot(iterations, labor_cost, color=colors['labor_cost'], linewidth=2, label='最优值')
        if 'avg_labor_cost' in convergence_data:
            ax2.plot(iterations, convergence_data['avg_labor_cost'],
                    color=colors['labor_cost'], linewidth=1, linestyle='--',
                    alpha=0.6, label='平均值')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('人工成本 (元)')
    ax2.set_title('人工成本 (F2)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy收敛曲线
    ax3 = axes[1, 0]
    energy = convergence_data.get('best_energy', [])
    if energy:
        ax3.plot(iterations, energy, color=colors['energy'], linewidth=2, label='最优值')
        if 'avg_energy' in convergence_data:
            ax3.plot(iterations, convergence_data['avg_energy'],
                    color=colors['energy'], linewidth=1, linestyle='--',
                    alpha=0.6, label='平均值')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('能耗 (kWh)')
    ax3.set_title('能源消耗 (F3)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Pareto解数量或综合信息
    ax4 = axes[1, 1]
    if 'n_pareto' in convergence_data:
        ax4.plot(iterations, convergence_data['n_pareto'], 
                color=colors['pareto'], linewidth=2)
        ax4.set_ylabel('Pareto解数量')
        ax4.set_title('Pareto前沿规模')
    elif 'archive_size' in convergence_data:
        ax4.plot(iterations, convergence_data['archive_size'],
                color=colors['pareto'], linewidth=2)
        ax4.set_ylabel('档案大小')
        ax4.set_title('Pareto档案规模')
    else:
        # 综合三个目标的趋势
        if makespan and labor_cost and energy:
            # 归一化后绘制
            norm_makespan = np.array(makespan) / (np.max(makespan) + 1e-10)
            norm_cost = np.array(labor_cost) / (np.max(labor_cost) + 1e-10)
            norm_energy = np.array(energy) / (np.max(energy) + 1e-10)
            
            ax4.plot(iterations, norm_makespan, color=colors['makespan'], 
                    linewidth=2, label='F1 (归一化)')
            ax4.plot(iterations, norm_cost, color=colors['labor_cost'], 
                    linewidth=2, label='F2 (归一化)')
            ax4.plot(iterations, norm_energy, color=colors['energy'], 
                    linewidth=2, label='F3 (归一化)')
            ax4.set_ylabel('归一化目标值')
            ax4.set_title('目标函数对比')
            ax4.legend(loc='upper right')
    
    ax4.set_xlabel('迭代次数')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(data_dict: Dict[str, Dict],
                    figsize: Tuple[int, int] = (14, 10),
                    save_path: Optional[str] = None) -> Figure:
    """
    绘制多算法对比收敛曲线
    
    Args:
        data_dict: {算法名: 收敛数据} 的字典
        figsize: 图像大小
        save_path: 保存路径
        
    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('多算法收敛对比', fontsize=14, fontweight='bold')
    
    # 颜色方案
    algorithm_colors = {
        'NSGA-II': '#2196F3',
        'VNS': '#4CAF50',
        'MOSA': '#FF5722'
    }
    
    objectives = ['best_makespan', 'best_labor_cost', 'best_energy']
    titles = ['最大完工时间 (F1)', '人工成本 (F2)', '能源消耗 (F3)']
    ylabels = ['Makespan (分钟)', '人工成本 (元)', '能耗 (kWh)']
    
    # 绘制三个目标的对比
    for idx, (obj, title, ylabel) in enumerate(zip(objectives, titles, ylabels)):
        ax = axes[idx // 2, idx % 2]
        
        for alg_name, data in data_dict.items():
            iterations = data.get('generation') or data.get('iteration', [])
            values = data.get(obj, [])
            
            if iterations and values:
                color = algorithm_colors.get(alg_name, f'C{list(data_dict.keys()).index(alg_name)}')
                ax.plot(iterations, values, color=color, linewidth=2, label=alg_name)
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # 第四个子图: Pareto解数量对比
    ax4 = axes[1, 1]
    for alg_name, data in data_dict.items():
        iterations = data.get('generation') or data.get('iteration', [])
        n_pareto = data.get('n_pareto') or data.get('archive_size', [])
        
        if iterations and n_pareto:
            color = algorithm_colors.get(alg_name, f'C{list(data_dict.keys()).index(alg_name)}')
            ax4.plot(iterations, n_pareto, color=color, linewidth=2, label=alg_name)
    
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('Pareto解数量')
    ax4.set_title('Pareto前沿规模')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
