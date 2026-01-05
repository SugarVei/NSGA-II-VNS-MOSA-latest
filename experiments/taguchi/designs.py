# -*- coding: utf-8 -*-
"""
田口实验设计模块
Taguchi Experiment Design Module

定义 L16(4^4) 正交表、因子水平映射和固定问题实例参数。
"""

import numpy as np
from typing import Dict, Any, List

# ==================== 因子与水平定义 ====================
# 4因子 × 4水平

FACTORS = {
    'A': {
        'name': 'population_size',
        'levels': [50, 100, 150, 200]
    },
    'B': {
        'name': 'crossover_prob',
        'levels': [0.70, 0.80, 0.90, 0.95]
    },
    'C': {
        'name': 'mutation_prob',
        'levels': [0.05, 0.10, 0.15, 0.20]
    },
    'D': {
        'name': 'initial_temp',
        'levels': [100, 300, 500, 1000]
    }
}

# ==================== L16(4^4) 正交表 ====================
# 16组实验，每组4个因子的水平索引 (0-3)
# 按用户给定的 Run1-Run16 顺序

L16_ARRAY = np.array([
    [0, 0, 0, 0],  # Run1:  1 1 1 1 -> 索引 0 0 0 0
    [0, 1, 1, 1],  # Run2:  1 2 2 2
    [0, 2, 2, 2],  # Run3:  1 3 3 3
    [0, 3, 3, 3],  # Run4:  1 4 4 4
    [1, 0, 1, 2],  # Run5:  2 1 2 3
    [1, 1, 0, 3],  # Run6:  2 2 1 4
    [1, 2, 3, 0],  # Run7:  2 3 4 1
    [1, 3, 2, 1],  # Run8:  2 4 3 2
    [2, 0, 2, 3],  # Run9:  3 1 3 4
    [2, 1, 3, 2],  # Run10: 3 2 4 3
    [2, 2, 0, 1],  # Run11: 3 3 1 2
    [2, 3, 1, 0],  # Run12: 3 4 2 1
    [3, 0, 3, 1],  # Run13: 4 1 4 2
    [3, 1, 2, 0],  # Run14: 4 2 3 1
    [3, 2, 1, 3],  # Run15: 4 3 2 4
    [3, 3, 0, 2],  # Run16: 4 4 1 3
], dtype=int)

# ==================== 固定问题实例参数 ====================
# 用于田口实验的真实案例参数

REAL_CASE_PARAMS = {
    'n_jobs': 15,
    'n_stages': 3,
    'machines_per_stage': [3, 3, 3],
    'n_speed_levels': 3,
    'n_skill_levels': 3,
    'workers_available': [7, 5, 3],  # A=7, B=5, C=3
}

# ==================== HYBRID 算法固定参数 ====================
# 这些参数在田口实验中保持固定

FIXED_HYBRID_PARAMS = {
    'n_generations': 100,      # NSGA-II 代数
    'cooling_rate': 0.95,      # MOSA 冷却系数
    'final_temp': 1.0,         # MOSA 终止温度
    'max_iterations': 50,      # MOSA 最大迭代
    'vns_iterations': 10,      # VNS 迭代次数
    'n_representative': 10,    # MOSA 代表解数量
}


def get_params_for_run(run_id: int) -> Dict[str, Any]:
    """
    获取指定运行编号的参数配置
    
    Args:
        run_id: 运行编号 (0-15)
        
    Returns:
        包含所有因子参数值的字典
    """
    if run_id < 0 or run_id >= len(L16_ARRAY):
        raise ValueError(f"run_id 必须在 0-{len(L16_ARRAY)-1} 范围内，得到 {run_id}")
    
    levels = L16_ARRAY[run_id]
    
    params = {
        'population_size': FACTORS['A']['levels'][levels[0]],
        'crossover_prob': FACTORS['B']['levels'][levels[1]],
        'mutation_prob': FACTORS['C']['levels'][levels[2]],
        'initial_temp': FACTORS['D']['levels'][levels[3]],
    }
    
    # 添加固定参数
    params.update(FIXED_HYBRID_PARAMS)
    
    return params


def get_level_indices_for_run(run_id: int) -> Dict[str, int]:
    """
    获取指定运行编号的水平索引 (用于分析)
    
    Args:
        run_id: 运行编号 (0-15)
        
    Returns:
        包含各因子水平索引的字典 (索引从1开始，符合田口惯例)
    """
    if run_id < 0 or run_id >= len(L16_ARRAY):
        raise ValueError(f"run_id 必须在 0-{len(L16_ARRAY)-1} 范围内，得到 {run_id}")
    
    levels = L16_ARRAY[run_id]
    
    return {
        'A_level': levels[0] + 1,  # 转为1-based
        'B_level': levels[1] + 1,
        'C_level': levels[2] + 1,
        'D_level': levels[3] + 1,
    }


def get_all_runs() -> List[Dict[str, Any]]:
    """
    获取所有16组运行的参数配置
    
    Returns:
        参数配置列表
    """
    return [get_params_for_run(i) for i in range(len(L16_ARRAY))]


def get_factor_names() -> List[str]:
    """获取因子名称列表"""
    return ['A', 'B', 'C', 'D']


def get_factor_param_names() -> List[str]:
    """获取因子对应的参数名称列表"""
    return [FACTORS[f]['name'] for f in get_factor_names()]


def get_n_levels() -> int:
    """获取水平数量"""
    return 4


def get_n_runs() -> int:
    """获取运行次数"""
    return len(L16_ARRAY)
