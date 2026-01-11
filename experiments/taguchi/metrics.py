# -*- coding: utf-8 -*-
"""
指标计算模块
Metrics Calculation Module

使用 pymoo 计算 IGD/GD/HV 指标。
"""

import numpy as np
from typing import Tuple, Dict, Any

# pymoo 指标导入
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV


def normalize_objectives(points: np.ndarray, 
                         f_min: np.ndarray, 
                         f_max: np.ndarray) -> np.ndarray:
    """
    Min-Max 归一化目标点
    
    Args:
        points: 目标点数组 (n, 3)
        f_min: 各目标最小值 (3,)
        f_max: 各目标最大值 (3,)
        
    Returns:
        归一化后的点 (n, 3)，值域 [0, 1]
    """
    if len(points) == 0:
        return points.copy()
    
    # 避免除零
    range_vals = f_max - f_min
    range_vals = np.where(range_vals < 1e-12, 1e-12, range_vals)
    
    normalized = (points - f_min) / range_vals
    
    # 裁剪到 [0, 1] 范围（处理超出 PF_ref 范围的点）
    normalized = np.clip(normalized, 0.0, None)
    
    return normalized


def get_hv_ref_point(pf_ref_norm: np.ndarray, margin: float = 0.5) -> np.ndarray:
    """
    计算 HV 参考点
    
    参考点设置为归一化 PF_ref 各目标最大值的 (1 + margin) 倍
    
    Args:
        pf_ref_norm: 归一化后的参考前沿 (n, 3)
        margin: 边距比例，默认 0.5 (即 50%) - 增大容差以防止 HV=0
        
    Returns:
        参考点 (3,)
    """
    if len(pf_ref_norm) == 0:
        return np.array([1.5, 1.5, 1.5])
    
    max_vals = np.max(pf_ref_norm, axis=0)
    ref_point = max_vals * (1.0 + margin)
    
    return ref_point


def compute_igd(A_norm: np.ndarray, pf_ref_norm: np.ndarray) -> float:
    """
    计算 Inverted Generational Distance (IGD)
    
    IGD 衡量参考前沿到解集的平均距离，越小越好。
    
    Args:
        A_norm: 归一化后的解集 (n, 3)
        pf_ref_norm: 归一化后的参考前沿 (m, 3)
        
    Returns:
        IGD 值
    """
    if len(A_norm) == 0 or len(pf_ref_norm) == 0:
        return float('inf')
    
    indicator = IGD(pf_ref_norm)
    return indicator.do(A_norm)


def compute_gd(A_norm: np.ndarray, pf_ref_norm: np.ndarray) -> float:
    """
    计算 Generational Distance (GD)
    
    GD 衡量解集到参考前沿的平均距离，越小越好。
    
    Args:
        A_norm: 归一化后的解集 (n, 3)
        pf_ref_norm: 归一化后的参考前沿 (m, 3)
        
    Returns:
        GD 值
    """
    if len(A_norm) == 0 or len(pf_ref_norm) == 0:
        return float('inf')
    
    indicator = GD(pf_ref_norm)
    return indicator.do(A_norm)


def compute_hv(A_norm: np.ndarray, ref_point: np.ndarray) -> float:
    """
    计算 Hypervolume (HV)
    
    HV 衡量解集支配的超体积，越大越好。
    
    Args:
        A_norm: 归一化后的解集 (n, 3)
        ref_point: 参考点 (3,)
        
    Returns:
        HV 值
    """
    if len(A_norm) == 0:
        return 0.0
    
    # 过滤掉超出参考点的解
    valid_mask = np.all(A_norm < ref_point, axis=1)
    A_valid = A_norm[valid_mask]
    
    # 兜底方案：如果所有解都超出参考点，自动扩展参考点
    # 这确保了 HV 不会因为解质量差而直接返回 0
    if len(A_valid) == 0:
        # 使用当前解集的最大值 * 1.1 作为新的参考点
        adaptive_ref_point = np.max(A_norm, axis=0) * 1.1
        A_valid = A_norm  # 使用所有解
        indicator = HV(ref_point=adaptive_ref_point)
        return indicator.do(A_valid)
    
    indicator = HV(ref_point=ref_point)
    return indicator.do(A_valid)


def compute_all_metrics(A: np.ndarray, 
                        pf_ref: np.ndarray,
                        f_min: np.ndarray,
                        f_max: np.ndarray,
                        hv_ref_point: np.ndarray) -> Dict[str, float]:
    """
    计算所有指标
    
    Args:
        A: 解集目标点 (n, 3)
        pf_ref: 参考前沿 (m, 3)
        f_min: 归一化最小值
        f_max: 归一化最大值
        hv_ref_point: HV 参考点（归一化空间）
        
    Returns:
        包含 igd, gd, hv 的字典
    """
    # 归一化
    A_norm = normalize_objectives(A, f_min, f_max)
    pf_ref_norm = normalize_objectives(pf_ref, f_min, f_max)
    
    return {
        'igd': compute_igd(A_norm, pf_ref_norm),
        'gd': compute_gd(A_norm, pf_ref_norm),
        'hv': compute_hv(A_norm, hv_ref_point),
    }


def get_normalization_info(pf_ref: np.ndarray, 
                           hv_margin: float = 0.5) -> Dict[str, Any]:
    """
    获取完整的归一化信息
    
    Args:
        pf_ref: 参考前沿 (n, 3)
        hv_margin: HV 参考点边距，默认 0.5 (增大以防止 HV=0)
        
    Returns:
        包含 f_min, f_max, hv_ref_point 的字典
    """
    if len(pf_ref) == 0:
        return {
            'f_min': [0.0, 0.0, 0.0],
            'f_max': [1.0, 1.0, 1.0],
            'hv_ref_point': [1.5, 1.5, 1.5],
        }
    
    f_min = np.min(pf_ref, axis=0)
    f_max = np.max(pf_ref, axis=0)
    
    # 在归一化空间计算 HV 参考点
    pf_ref_norm = normalize_objectives(pf_ref, f_min, f_max)
    hv_ref_point = get_hv_ref_point(pf_ref_norm, hv_margin)
    
    return {
        'f_min': f_min.tolist(),
        'f_max': f_max.tolist(),
        'hv_ref_point': hv_ref_point.tolist(),
    }
