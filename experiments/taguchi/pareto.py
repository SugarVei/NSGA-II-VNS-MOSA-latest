# -*- coding: utf-8 -*-
"""
Pareto 前沿模块
Pareto Front Module

非支配筛选和 PF_ref 构造功能。
"""

import numpy as np
from typing import List, Tuple


def is_dominated(p: np.ndarray, q: np.ndarray) -> bool:
    """
    判断点 p 是否被点 q 支配（三目标最小化问题）
    
    p 被 q 支配 当且仅当:
    1. q 在所有目标上都不比 p 差 (q <= p)
    2. q 在至少一个目标上严格优于 p (q < p)
    
    Args:
        p: 被检查的点 (3,)
        q: 用于支配检查的点 (3,)
        
    Returns:
        True 如果 p 被 q 支配
    """
    return np.all(q <= p) and np.any(q < p)


def get_non_dominated(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从点集中筛选非支配点
    
    Args:
        points: 目标点数组 (n_points, 3)
        
    Returns:
        (non_dominated_points, indices): 非支配点数组和对应的原始索引
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 3), np.array([], dtype=int)
    
    n = len(points)
    is_non_dominated = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_non_dominated[i]:
            continue
        for j in range(n):
            if i == j or not is_non_dominated[j]:
                continue
            # 如果 i 被 j 支配
            if is_dominated(points[i], points[j]):
                is_non_dominated[i] = False
                break
    
    indices = np.where(is_non_dominated)[0]
    return points[is_non_dominated].copy(), indices


def build_pf_ref(all_objectives: List[np.ndarray]) -> np.ndarray:
    """
    构造全局参考前沿 PF_ref
    
    将多次运行得到的所有非支配解目标点合并，然后做全局非支配筛选。
    
    Args:
        all_objectives: 每次运行的目标点数组列表，每个元素形状 (n_i, 3)
        
    Returns:
        PF_ref: 全局非支配前沿 (n_pf, 3)
    """
    if not all_objectives:
        return np.array([]).reshape(0, 3)
    
    # 合并所有目标点
    combined = np.vstack([obj for obj in all_objectives if len(obj) > 0])
    
    if len(combined) == 0:
        return np.array([]).reshape(0, 3)
    
    # 全局非支配筛选
    pf_ref, _ = get_non_dominated(combined)
    
    return pf_ref


def compute_normalization_params(pf_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算归一化参数
    
    Args:
        pf_ref: 参考前沿 (n_pf, 3)
        
    Returns:
        (f_min, f_max): 各目标的最小值和最大值
    """
    if len(pf_ref) == 0:
        return np.zeros(3), np.ones(3)
    
    f_min = np.min(pf_ref, axis=0)
    f_max = np.max(pf_ref, axis=0)
    
    return f_min, f_max


def filter_dominated_from_archive(archive: np.ndarray, new_point: np.ndarray) -> np.ndarray:
    """
    将新点加入档案并移除被支配的点
    
    Args:
        archive: 现有档案 (n, 3)
        new_point: 新点 (3,)
        
    Returns:
        更新后的档案
    """
    if len(archive) == 0:
        return new_point.reshape(1, 3)
    
    # 检查新点是否被档案中任何点支配
    for i in range(len(archive)):
        if is_dominated(new_point, archive[i]):
            return archive  # 新点被支配，不添加
    
    # 移除被新点支配的点
    keep_mask = np.array([not is_dominated(archive[i], new_point) 
                          for i in range(len(archive))])
    
    new_archive = archive[keep_mask]
    new_archive = np.vstack([new_archive, new_point.reshape(1, 3)])
    
    return new_archive
