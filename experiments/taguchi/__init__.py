# -*- coding: utf-8 -*-
"""
田口实验模块
Taguchi Experiment Module

用于 HYBRID 算法 (NSGA-II + VNS + MOSA) 的参数调优实验。
"""

# 基础模块（无外部依赖）
from .designs import FACTORS, L16_ARRAY, get_params_for_run, REAL_CASE_PARAMS
from .pareto import get_non_dominated, build_pf_ref
from .io import save_csv, save_json, ErrorLogger

# pymoo 依赖的模块使用延迟导入
def _check_pymoo():
    """检查 pymoo 是否已安装"""
    try:
        import pymoo
        return True
    except ImportError:
        return False

__all__ = [
    'FACTORS', 'L16_ARRAY', 'get_params_for_run', 'REAL_CASE_PARAMS',
    'get_non_dominated', 'build_pf_ref',
    'save_csv', 'save_json', 'ErrorLogger',
]

# 尝试导入依赖 pymoo 的模块
if _check_pymoo():
    from .metrics import (
        normalize_objectives, compute_igd, compute_gd, compute_hv
    )
    from .analysis import (
        compute_snr_smaller_better, compute_snr_larger_better,
        compute_summary_statistics, generate_paper_summary
    )
    from .plotting import plot_main_effects, plot_boxplot_by_run, plot_pareto_projections
    
    __all__.extend([
        'normalize_objectives', 'compute_igd', 'compute_gd', 'compute_hv',
        'compute_snr_smaller_better', 'compute_snr_larger_better',
        'compute_summary_statistics', 'generate_paper_summary',
        'plot_main_effects', 'plot_boxplot_by_run', 'plot_pareto_projections',
    ])

