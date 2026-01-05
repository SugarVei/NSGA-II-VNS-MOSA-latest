# -*- coding: utf-8 -*-
"""
算例配置数据模块
Case Configuration Data Module

定义算例配置数据结构、默认14个算例、JSON持久化功能。
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class CaseConfig:
    """
    单个算例配置
    
    Attributes:
        case_no: 序号
        n_jobs: 工件数
        machines_per_stage: 各阶段机器数 [3 items for 3 stages]
        workers_available: 各技能等级工人数 [3 items]
        input_mode: 数据输入模式 ('auto' | 'manual' | None)
        problem_data: 实例数据字典（processing_time, setup_time 等）
        algorithm_params: 各算法参数覆盖
        is_configured: 是否已配置完成
    """
    case_no: int
    n_jobs: int
    machines_per_stage: List[int]  # 3个阶段
    workers_available: List[int]   # 3个技能等级
    input_mode: Optional[str] = None
    problem_data: Optional[Dict[str, Any]] = None
    algorithm_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    is_configured: bool = False
    
    @property
    def total_machines(self) -> int:
        """总机器数"""
        return sum(self.machines_per_stage)
    
    @property
    def total_workers(self) -> int:
        """总工人数"""
        return sum(self.workers_available)
    
    @property
    def problem_scale_str(self) -> str:
        """问题规模字符串，如 '7×20×8'"""
        return f"{self.total_machines}×{self.n_jobs}×{self.total_workers}"
    
    @property
    def machines_dist_str(self) -> str:
        """机器分布字符串，如 '2×3×2'"""
        return "×".join(str(m) for m in self.machines_per_stage)
    
    @property
    def workers_dist_str(self) -> str:
        """工人分布字符串，如 '4×3×1'"""
        return "×".join(str(w) for w in self.workers_available)
    
    def to_dict(self) -> dict:
        """转换为可序列化字典"""
        d = {
            'case_no': self.case_no,
            'n_jobs': self.n_jobs,
            'machines_per_stage': self.machines_per_stage,
            'workers_available': self.workers_available,
            'input_mode': self.input_mode,
            'is_configured': self.is_configured,
            'algorithm_params': self.algorithm_params,
        }
        # problem_data 中的 numpy 数组需要转换
        if self.problem_data:
            d['problem_data'] = {}
            for key, val in self.problem_data.items():
                if isinstance(val, np.ndarray):
                    d['problem_data'][key] = val.tolist()
                else:
                    d['problem_data'][key] = val
        else:
            d['problem_data'] = None
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'CaseConfig':
        """从字典创建实例"""
        problem_data = d.get('problem_data')
        if problem_data:
            # 将 list 转回 numpy 数组
            for key in ['processing_time', 'setup_time', 'transport_time',
                        'processing_power', 'setup_power', 'idle_power',
                        'skill_wages', 'skill_compatibility', 'workers_available_arr']:
                if key in problem_data and problem_data[key] is not None:
                    problem_data[key] = np.array(problem_data[key])
        
        return cls(
            case_no=d['case_no'],
            n_jobs=d['n_jobs'],
            machines_per_stage=d['machines_per_stage'],
            workers_available=d['workers_available'],
            input_mode=d.get('input_mode'),
            problem_data=problem_data,
            algorithm_params=d.get('algorithm_params', {}),
            is_configured=d.get('is_configured', False)
        )


# ========== 默认8个算法的参数（田口实验最优参数） ==========
# 参数说明：N=种群大小, Pc=交叉概率, Pm=变异率
# 对于未在田口实验表中涉及的参数，保持默认值
DEFAULT_ALGORITHM_PARAMS = {
    # NSGA-II: N=200, Pc=0.95, Pm=0.15
    'NSGA-II': {
        'pop_size': 200,
        'n_generations': 200,         # 对比算法减少迭代以加速
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
    },
    # MOSA: T0=100, max iterations=200, α=0.95, L=100
    'MOSA': {
        'initial_temp': 100.0,        # 田口最优: T0=100
        'cooling_rate': 0.95,         # 田口最优: α=0.95
        'final_temp': 0.001,          # 默认值
        'max_iterations': 200,        # 对比算法减少迭代以加速
        'markov_chain_length': 100,   # 田口最优: L=100
        'n_representative': 5,        # 默认值（未在表中指定）
    },
    # MOEA/D: N=200, Pc=0.95, Pm=0.15, T=40, weight vectors=N
    'MOEA/D': {
        'pop_size': 200,
        'n_generations': 200,         # 对比算法减少迭代以加速
        'neighborhood_size': 40,      # 田口最优: T=40
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
    },
    # SPEA2: N=200, Pc=0.95, Pm=0.15, archive size=100
    'SPEA2': {
        'pop_size': 200,
        'n_generations': 200,         # 对比算法减少迭代以加速
        'archive_size': 100,          # 田口最优: archive size=100
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
    },
    # MOPSO: swarm size N=200, max iterations=200, w=0.5, c1=c2=1.5
    'MOPSO': {
        'swarm_size': 200,
        'max_iterations': 200,        # 对比算法减少迭代以加速
        'w': 0.5,                     # 田口最优: inertia weight w=0.5
        'c1': 1.5,                    # 田口最优: c1=1.5
        'c2': 1.5,                    # 田口最优: c2=1.5
        'repository_size': 200,       # 默认值（未在表中指定）
        'mutation_prob': 0.1,         # 默认值（未在表中指定）
    },
    # NSGA-II-VNS: N=200, Pc=0.95, Pm=0.15, VNS neighborhood structures=4
    'NSGA2-VNS': {
        'pop_size': 200,
        'n_generations': 200,         # 对比算法减少迭代以加速
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
        'vns_neighborhood_structures': 4,  # 田口最优: VNS neighborhood structures=4
    },
    # NSGA-II-MOSA: N=200, Pc=0.95, Pm=0.15, T0=1000, α=0.95, L=100
    'NSGA2-MOSA': {
        'pop_size': 200,
        'n_generations': 200,         # 对比算法减少迭代以加速
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
        'initial_temp': 1000.0,       # 田口最优: T0=1000
        'cooling_rate': 0.95,         # 田口最优: α=0.95
        'markov_chain_length': 100,   # 田口最优: L=100
    },
    # NSGA-II-VNS-MOSA: N=200, Pc=0.95, Pm=0.15, T0=1000, VNS=4, α=0.95, L=100
    'NSGA2-VNS-MOSA': {
        'pop_size': 200,
        'n_generations': 500,
        'crossover_prob': 0.95,       # 田口最优: Pc=0.95
        'mutation_prob': 0.15,        # 田口最优: Pm=0.15
        'initial_temp': 1000.0,       # 田口最优: T0=1000
        'cooling_rate': 0.95,         # 田口最优: α=0.95
        'markov_chain_length': 100,   # 田口最优: L=100
        'vns_neighborhood_structures': 4,  # 田口最优: VNS neighborhood structures=4
    },
}


def get_default_algorithm_params() -> Dict[str, Dict[str, Any]]:
    """获取默认算法参数的深拷贝"""
    import copy
    return copy.deepcopy(DEFAULT_ALGORITHM_PARAMS)


# ========== 默认14个算例数据 ==========
DEFAULT_CASES_DATA = [
    # (case_no, n_jobs, machines_per_stage, workers_available)
    (1,  20,  [2, 3, 2], [4, 3, 1]),
    (2,  40,  [2, 3, 2], [4, 3, 1]),
    (3,  60,  [3, 4, 3], [5, 4, 2]),
    (4,  80,  [3, 4, 3], [5, 4, 2]),
    (5,  100, [3, 5, 3], [5, 4, 3]),
    (6,  120, [4, 5, 4], [6, 5, 3]),
    (7,  140, [4, 6, 4], [6, 5, 4]),
    (8,  160, [5, 6, 5], [8, 6, 4]),
    (9,  200, [5, 7, 5], [8, 6, 5]),
    (10, 240, [6, 7, 6], [9, 7, 5]),
    (11, 280, [6, 8, 6], [9, 7, 6]),
    (12, 320, [7, 8, 7], [10, 8, 6]),
    (13, 360, [7, 9, 7], [11, 8, 6]),
    (14, 400, [8, 9, 8], [11, 9, 7]),
]


def get_default_cases() -> List[CaseConfig]:
    """
    获取默认的14个算例配置
    
    Returns:
        CaseConfig 列表
    """
    cases = []
    for case_no, n_jobs, machines, workers in DEFAULT_CASES_DATA:
        case = CaseConfig(
            case_no=case_no,
            n_jobs=n_jobs,
            machines_per_stage=machines,
            workers_available=workers,
            algorithm_params=get_default_algorithm_params()
        )
        cases.append(case)
    return cases


# ========== JSON 持久化 ==========
def save_cases_config(cases: List[CaseConfig], filepath: str) -> bool:
    """
    保存算例配置到 JSON 文件
    
    Args:
        cases: CaseConfig 列表
        filepath: 目标文件路径
        
    Returns:
        是否成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'version': '1.0',
            'cases': [case.to_dict() for case in cases]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"保存算例配置失败: {e}")
        return False


def load_cases_config(filepath: str) -> Optional[List[CaseConfig]]:
    """
    从 JSON 文件加载算例配置
    
    Args:
        filepath: JSON 文件路径
        
    Returns:
        CaseConfig 列表，加载失败返回 None
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cases = [CaseConfig.from_dict(d) for d in data.get('cases', [])]
        return cases if cases else None
    except Exception as e:
        print(f"加载算例配置失败: {e}")
        return None


# 默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'comparison_cases.json'
)


def parse_distribution_string(s: str) -> Optional[List[int]]:
    """
    解析分布字符串，如 '2×3×2' -> [2, 3, 2]
    
    Args:
        s: 分布字符串，使用 × 或 x 分隔
        
    Returns:
        整数列表或 None（解析失败）
    """
    try:
        # 支持 × 和 x 两种分隔符
        s = s.replace('×', 'x').replace('X', 'x')
        parts = s.split('x')
        return [int(p.strip()) for p in parts]
    except:
        return None


def validate_case_config(case: CaseConfig) -> tuple:
    """
    验证算例配置的有效性
    
    Args:
        case: CaseConfig 实例
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if case.n_jobs <= 0:
        errors.append("工件数必须大于0")
    
    if len(case.machines_per_stage) != 3:
        errors.append("必须指定3个阶段的机器数")
    elif any(m <= 0 for m in case.machines_per_stage):
        errors.append("每阶段机器数必须大于0")
    
    if len(case.workers_available) != 3:
        errors.append("必须指定3个技能等级的工人数")
    elif any(w < 0 for w in case.workers_available):
        errors.append("工人数不能为负")
    
    return len(errors) == 0, errors
