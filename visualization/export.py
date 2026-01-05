"""
结果导出模块
Export Module

将优化结果导出为CSV文件和图像文件。
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution


def export_pareto_to_csv(solutions: List[Solution],
                         filepath: str,
                         include_decisions: bool = False) -> str:
    """
    将Pareto解集导出为CSV文件
    
    Args:
        solutions: Pareto解列表
        filepath: 输出文件路径
        include_decisions: 是否包含决策变量详情
        
    Returns:
        实际保存的文件路径
    """
    if not solutions:
        raise ValueError("解集为空，无法导出")
    
    # 基础目标值
    data = {
        '解编号': list(range(1, len(solutions) + 1)),
        'Makespan(分钟)': [s.objectives[0] for s in solutions],
        '人工成本(元)': [s.objectives[1] for s in solutions],
        '能耗(kWh)': [s.objectives[2] for s in solutions],
        'Pareto排名': [s.rank for s in solutions],
        '拥挤度': [s.crowding_distance for s in solutions]
    }
    
    if include_decisions:
        # 添加决策变量摘要
        for i, sol in enumerate(solutions):
            # 机器使用分布
            unique_machines = np.unique(sol.machine_assign)
            data.setdefault('使用机器数', []).append(len(unique_machines))
            
            # 平均速度
            avg_speed = np.mean(sol.speed_level)
            data.setdefault('平均速度等级', []).append(f'{avg_speed:.2f}')
            
            # 平均技能
            avg_skill = np.mean(sol.worker_skill)
            data.setdefault('平均工人技能', []).append(f'{avg_skill:.2f}')
    
    df = pd.DataFrame(data)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    return filepath


def export_convergence_to_csv(convergence_data: Dict,
                              algorithm_name: str,
                              filepath: str) -> str:
    """
    将收敛历史导出为CSV
    
    Args:
        convergence_data: 收敛历史数据
        algorithm_name: 算法名称
        filepath: 输出文件路径
        
    Returns:
        实际保存的文件路径
    """
    df = pd.DataFrame(convergence_data)
    df.insert(0, '算法', algorithm_name)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    return filepath


def generate_report(solutions: List[Solution],
                    convergence_data: Dict,
                    algorithm_name: str,
                    output_dir: str) -> Dict[str, str]:
    """
    生成完整的优化报告
    
    Args:
        solutions: Pareto解集
        convergence_data: 收敛历史
        algorithm_name: 算法名称
        output_dir: 输出目录
        
    Returns:
        生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files = {}
    
    # 导出Pareto解集
    pareto_path = os.path.join(output_dir, f'pareto_{algorithm_name}_{timestamp}.csv')
    export_pareto_to_csv(solutions, pareto_path, include_decisions=True)
    files['pareto_csv'] = pareto_path
    
    # 导出收敛历史
    if convergence_data:
        conv_path = os.path.join(output_dir, f'convergence_{algorithm_name}_{timestamp}.csv')
        export_convergence_to_csv(convergence_data, algorithm_name, conv_path)
        files['convergence_csv'] = conv_path
    
    # 生成摘要报告
    summary_path = os.path.join(output_dir, f'summary_{algorithm_name}_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*50}\n")
        f.write(f"多目标调度优化结果报告\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"算法: {algorithm_name}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Pareto解数量: {len(solutions)}\n\n")
        
        if solutions:
            objectives = np.array([s.objectives for s in solutions])
            
            f.write("目标函数统计:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Makespan (F1):\n")
            f.write(f"  最小值: {objectives[:, 0].min():.2f} 分钟\n")
            f.write(f"  最大值: {objectives[:, 0].max():.2f} 分钟\n")
            f.write(f"  平均值: {objectives[:, 0].mean():.2f} 分钟\n\n")
            
            f.write(f"人工成本 (F2):\n")
            f.write(f"  最小值: {objectives[:, 1].min():.2f} 元\n")
            f.write(f"  最大值: {objectives[:, 1].max():.2f} 元\n")
            f.write(f"  平均值: {objectives[:, 1].mean():.2f} 元\n\n")
            
            f.write(f"能耗 (F3):\n")
            f.write(f"  最小值: {objectives[:, 2].min():.2f} kWh\n")
            f.write(f"  最大值: {objectives[:, 2].max():.2f} kWh\n")
            f.write(f"  平均值: {objectives[:, 2].mean():.2f} kWh\n\n")
            
            # 找出各目标最优解
            best_makespan = solutions[np.argmin(objectives[:, 0])]
            best_cost = solutions[np.argmin(objectives[:, 1])]
            best_energy = solutions[np.argmin(objectives[:, 2])]
            
            f.write("各目标最优解:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Makespan最优: {best_makespan.objectives}\n")
            f.write(f"人工成本最优: {best_cost.objectives}\n")
            f.write(f"能耗最优: {best_energy.objectives}\n")
    
    files['summary_txt'] = summary_path
    
    return files
