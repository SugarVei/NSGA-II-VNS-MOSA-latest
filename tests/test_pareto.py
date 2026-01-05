# -*- coding: utf-8 -*-
"""
Pareto 模块单元测试
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.taguchi.pareto import (
    is_dominated, get_non_dominated, build_pf_ref
)


class TestIsDominated:
    """支配关系测试"""
    
    def test_dominated_all_worse(self):
        """p 在所有目标上都比 q 差"""
        p = np.array([5.0, 5.0, 5.0])
        q = np.array([3.0, 3.0, 3.0])
        assert is_dominated(p, q) == True
    
    def test_dominated_one_equal(self):
        """p 有一个目标与 q 相等，其余更差"""
        p = np.array([5.0, 3.0, 5.0])
        q = np.array([3.0, 3.0, 3.0])
        assert is_dominated(p, q) == True
    
    def test_not_dominated_one_better(self):
        """p 有一个目标比 q 好"""
        p = np.array([5.0, 2.0, 5.0])
        q = np.array([3.0, 3.0, 3.0])
        assert is_dominated(p, q) == False
    
    def test_not_dominated_equal(self):
        """p 和 q 完全相等"""
        p = np.array([3.0, 3.0, 3.0])
        q = np.array([3.0, 3.0, 3.0])
        assert is_dominated(p, q) == False
    
    def test_not_dominated_all_better(self):
        """p 在所有目标上都比 q 好"""
        p = np.array([2.0, 2.0, 2.0])
        q = np.array([3.0, 3.0, 3.0])
        assert is_dominated(p, q) == False


class TestGetNonDominated:
    """非支配筛选测试"""
    
    def test_empty_input(self):
        """空输入"""
        points = np.array([]).reshape(0, 3)
        result, indices = get_non_dominated(points)
        assert len(result) == 0
        assert len(indices) == 0
    
    def test_single_point(self):
        """单点"""
        points = np.array([[1.0, 2.0, 3.0]])
        result, indices = get_non_dominated(points)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
    
    def test_all_non_dominated(self):
        """所有点都非支配（Pareto 前沿）"""
        points = np.array([
            [1.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 1.0],
        ])
        result, indices = get_non_dominated(points)
        assert len(result) == 3
    
    def test_one_dominated(self):
        """有一个点被支配"""
        points = np.array([
            [1.0, 1.0, 1.0],  # 支配点
            [2.0, 2.0, 2.0],  # 被支配
            [1.0, 2.0, 3.0],  # 非支配
        ])
        result, indices = get_non_dominated(points)
        assert len(result) == 2
        assert 1 not in indices
    
    def test_multiple_dominated(self):
        """多个点被支配"""
        points = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ])
        result, indices = get_non_dominated(points)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])


class TestBuildPfRef:
    """PF_ref 构造测试"""
    
    def test_empty_input(self):
        """空输入"""
        result = build_pf_ref([])
        assert len(result) == 0
    
    def test_single_run(self):
        """单次运行"""
        objectives = [np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])]
        result = build_pf_ref(objectives)
        assert len(result) == 2
    
    def test_multiple_runs_merge(self):
        """多次运行合并"""
        objectives = [
            np.array([[1.0, 5.0, 5.0]]),
            np.array([[5.0, 1.0, 5.0]]),
            np.array([[5.0, 5.0, 1.0]]),
        ]
        result = build_pf_ref(objectives)
        assert len(result) == 3
    
    def test_dominated_removed_after_merge(self):
        """合并后移除被支配点"""
        objectives = [
            np.array([[1.0, 1.0, 1.0]]),
            np.array([[2.0, 2.0, 2.0]]),
        ]
        result = build_pf_ref(objectives)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 1.0, 1.0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
