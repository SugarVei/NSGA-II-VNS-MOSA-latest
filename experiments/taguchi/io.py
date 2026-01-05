# -*- coding: utf-8 -*-
"""
IO 模块
Input/Output Module

CSV/JSON 保存、加载和异常记录。
"""

import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON 编码器，支持 NumPy 类型"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    保存 DataFrame 到 CSV
    
    Args:
        df: 要保存的 DataFrame
        path: 保存路径
        index: 是否保存索引
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    保存字典到 JSON
    
    Args:
        data: 要保存的字典
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    从 JSON 加载字典
    
    Args:
        path: JSON 文件路径
        
    Returns:
        加载的字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    """
    从 CSV 加载 DataFrame
    
    Args:
        path: CSV 文件路径
        
    Returns:
        加载的 DataFrame
    """
    return pd.read_csv(path, encoding='utf-8-sig')


class ErrorLogger:
    """
    异常记录器
    
    捕获并记录实验运行中的异常，确保单次失败不影响整体实验。
    """
    
    def __init__(self, log_path: Path):
        """
        初始化异常记录器
        
        Args:
            log_path: 日志文件路径
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.errors: List[Dict[str, Any]] = []
        
        # 写入头部
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Taguchi Experiment Error Log\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"# ================================\n\n")
    
    def log_error(self, 
                  run_id: int, 
                  rep_id: int, 
                  params: Dict[str, Any], 
                  seed: int, 
                  error: Exception) -> None:
        """
        记录一次运行的异常
        
        Args:
            run_id: 运行编号
            rep_id: 重复编号
            params: 参数配置
            seed: 随机种子
            error: 异常对象
        """
        error_info = {
            'run_id': run_id,
            'rep_id': rep_id,
            'params': params,
            'seed': seed,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
        }
        self.errors.append(error_info)
        
        # 追加写入日志文件
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n## Error at Run {run_id}, Rep {rep_id}\n")
            f.write(f"- **Time**: {error_info['timestamp']}\n")
            f.write(f"- **Seed**: {seed}\n")
            f.write(f"- **Params**: {params}\n")
            f.write(f"- **Error Type**: {error_info['error_type']}\n")
            f.write(f"- **Message**: {error_info['error_message']}\n")
            f.write(f"- **Traceback**:\n```\n{traceback.format_exc()}\n```\n")
    
    def get_error_count(self) -> int:
        """获取错误数量"""
        return len(self.errors)
    
    def get_failed_runs(self) -> List[tuple]:
        """获取失败的 (run_id, rep_id) 列表"""
        return [(e['run_id'], e['rep_id']) for e in self.errors]
    
    def finalize(self) -> None:
        """完成日志记录"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n# ================================\n")
            f.write(f"# Summary\n")
            f.write(f"# Total Errors: {len(self.errors)}\n")
            f.write(f"# Completed: {datetime.now().isoformat()}\n")


def save_pf_ref(pf_ref: np.ndarray, path: Path) -> None:
    """
    保存参考前沿到 CSV
    
    Args:
        pf_ref: 参考前沿 (n, 3)
        path: 保存路径
    """
    df = pd.DataFrame(pf_ref, columns=['f1', 'f2', 'f3'])
    save_csv(df, path)


def load_pf_ref(path: Path) -> np.ndarray:
    """
    从 CSV 加载参考前沿
    
    Args:
        path: CSV 文件路径
        
    Returns:
        参考前沿数组 (n, 3)
    """
    df = load_csv(path)
    return df[['f1', 'f2', 'f3']].values


def save_normalization(f_min: np.ndarray, 
                       f_max: np.ndarray, 
                       path: Path) -> None:
    """
    保存归一化参数
    
    Args:
        f_min: 最小值 (3,)
        f_max: 最大值 (3,)
        path: 保存路径
    """
    data = {
        'f_min': f_min.tolist() if isinstance(f_min, np.ndarray) else f_min,
        'f_max': f_max.tolist() if isinstance(f_max, np.ndarray) else f_max,
    }
    save_json(data, path)


def save_hv_ref_point(ref_point: np.ndarray, path: Path) -> None:
    """
    保存 HV 参考点
    
    Args:
        ref_point: 参考点 (3,)
        path: 保存路径
    """
    data = {
        'ref_point': ref_point.tolist() if isinstance(ref_point, np.ndarray) else ref_point,
    }
    save_json(data, path)


def create_output_structure(base_dir: Path) -> Dict[str, Path]:
    """
    创建输出目录结构
    
    Args:
        base_dir: 基础输出目录
        
    Returns:
        各输出路径的字典
    """
    base_dir = Path(base_dir)
    figures_dir = base_dir / 'figures'
    
    base_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': base_dir,
        'figures': figures_dir,
        'taguchi_runs': base_dir / 'taguchi_runs.csv',
        'taguchi_summary': base_dir / 'taguchi_summary.csv',
        'taguchi_summary_paper': base_dir / 'taguchi_summary_paper.csv',
        'pf_ref': base_dir / 'pf_ref.csv',
        'normalization': base_dir / 'normalization.json',
        'hv_ref_point': base_dir / 'hv_ref_point.json',
        'snr_effect_igd': base_dir / 'snr_effect_igd.csv',
        'snr_effect_hv': base_dir / 'snr_effect_hv.csv',
        'snr_effect_gd': base_dir / 'snr_effect_gd.csv',
        'best_params': base_dir / 'best_params.json',
        'confirm_summary': base_dir / 'confirm_experiment_summary.csv',
        'error_log': base_dir / 'error_log.txt',
    }
