"""
3D Pareto前沿可视化模块
3D Pareto Front Visualization

绘制三目标优化的Pareto前沿3D散点图。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_pareto_3d(solutions: List[Solution],
                   figsize: Tuple[int, int] = (10, 8),
                   title: str = "Pareto前沿",
                   save_path: Optional[str] = None,
                   elev: int = 20,
                   azim: int = 45,
                   weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Figure:
    """
    绘制3D Pareto前沿散点图
    
    Args:
        solutions: Pareto解列表
        figsize: 图像大小
        title: 图表标题
        save_path: 保存路径
        elev: 俯仰角
        azim: 方位角
        weights: 目标权重 (w1, w2, w3) 用于计算综合目标值
        
    Returns:
        matplotlib Figure对象
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if not solutions:
        ax.set_title("无Pareto解")
        return fig
    
    # 提取目标值
    objectives = np.array([s.objectives for s in solutions if s.objectives is not None])
    
    if len(objectives) == 0:
        ax.set_title("无有效目标值")
        return fig
    
    makespan = objectives[:, 0]
    labor_cost = objectives[:, 1]
    energy = objectives[:, 2]
    
    # 计算综合目标值 (加权归一化和)
    # 使用 min-max 归一化避免除零问题
    def safe_normalize(arr):
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    norm_makespan = safe_normalize(makespan)
    norm_cost = safe_normalize(labor_cost)
    norm_energy = safe_normalize(energy)
    
    # 加权求和: 综合目标值 = w1*F1_norm + w2*F2_norm + w3*F3_norm
    w1, w2, w3 = weights
    weighted_sum = w1 * norm_makespan + w2 * norm_cost + w3 * norm_energy
    
    # 绘制3D散点
    scatter = ax.scatter(makespan, labor_cost, energy,
                        c=weighted_sum,
                        cmap='viridis',
                        s=80,
                        alpha=0.8,
                        edgecolors='white',
                        linewidths=0.5)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(f'综合目标值 (权重 {w1}:{w2}:{w3})', fontsize=10)
    
    # 设置标签
    ax.set_xlabel('Makespan (分钟)', fontsize=11, labelpad=10)
    ax.set_ylabel('人工成本 (元)', fontsize=11, labelpad=10)
    ax.set_zlabel('能耗 (kWh)', fontsize=11, labelpad=10)
    
    ax.set_title(f'{title}\n(共 {len(objectives)} 个Pareto解)', fontsize=12, fontweight='bold')
    
    # 设置视角
    ax.view_init(elev=elev, azim=azim)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pareto_2d_projections(solutions: List[Solution],
                                figsize: Tuple[int, int] = (14, 5),
                                save_path: Optional[str] = None) -> Figure:
    """
    绘制Pareto前沿的2D投影图
    
    Args:
        solutions: Pareto解列表
        figsize: 图像大小
        save_path: 保存路径
        
    Returns:
        matplotlib Figure对象
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Pareto前沿 2D投影', fontsize=14, fontweight='bold')
    
    if not solutions:
        return fig
    
    objectives = np.array([s.objectives for s in solutions if s.objectives is not None])
    
    if len(objectives) == 0:
        return fig
    
    makespan = objectives[:, 0]
    labor_cost = objectives[:, 1]
    energy = objectives[:, 2]
    
    # 颜色
    color = '#2196F3'
    
    # 1. Makespan vs Labor Cost
    axes[0].scatter(makespan, labor_cost, c=color, s=60, alpha=0.7, edgecolors='white')
    axes[0].set_xlabel('Makespan (分钟)')
    axes[0].set_ylabel('人工成本 (元)')
    axes[0].set_title('F1 vs F2')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Makespan vs Energy
    axes[1].scatter(makespan, energy, c=color, s=60, alpha=0.7, edgecolors='white')
    axes[1].set_xlabel('Makespan (分钟)')
    axes[1].set_ylabel('能耗 (kWh)')
    axes[1].set_title('F1 vs F3')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Labor Cost vs Energy
    axes[2].scatter(labor_cost, energy, c=color, s=60, alpha=0.7, edgecolors='white')
    axes[2].set_xlabel('人工成本 (元)')
    axes[2].set_ylabel('能耗 (kWh)')
    axes[2].set_title('F2 vs F3')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pareto_comparison(solutions_dict: dict,
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None) -> Figure:
    """
    绘制多算法Pareto前沿对比图
    
    Args:
        solutions_dict: {算法名: 解列表} 的字典
        figsize: 图像大小
        save_path: 保存路径
        
    Returns:
        matplotlib Figure对象
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (alg_name, solutions) in enumerate(solutions_dict.items()):
        if not solutions:
            continue
        
        objectives = np.array([s.objectives for s in solutions if s.objectives is not None])
        
        if len(objectives) == 0:
            continue
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                  c=color, marker=marker, s=60, alpha=0.7,
                  label=f'{alg_name} ({len(objectives)}个解)',
                  edgecolors='white', linewidths=0.5)
    
    ax.set_xlabel('Makespan (分钟)', fontsize=11, labelpad=10)
    ax.set_ylabel('人工成本 (元)', fontsize=11, labelpad=10)
    ax.set_zlabel('能耗 (kWh)', fontsize=11, labelpad=10)
    
    ax.set_title('多算法Pareto前沿对比', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
