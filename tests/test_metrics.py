# -*- coding: utf-8 -*-
"""
指标计算模块单元测试
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.taguchi.metrics import (
    normalize_objectives, get_hv_ref_point,
    compute_igd, compute_gd, compute_hv, compute_all_metrics
)


class TestNormalizeObjectives:
    """归一化测试"""
    
    def test_basic_normalization(self):
        """基本归一化"""
        points = np.array([[0.0, 50.0, 100.0], [100.0, 100.0, 200.0]])
        f_min = np.array([0.0, 0.0, 0.0])
        f_max = np.array([100.0, 100.0, 200.0])
        
        result = normalize_objectives(points, f_min, f_max)
        
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result[1], [1.0, 1.0, 1.0])
    
    def test_empty_input(self):
        """空输入"""
        points = np.array([]).reshape(0, 3)
        f_min = np.array([0.0, 0.0, 0.0])
        f_max = np.array([1.0, 1.0, 1.0])
        
        result = normalize_objectives(points, f_min, f_max)
        
        assert len(result) == 0
    
    def test_zero_range(self):
        """范围为零时避免除零"""
        points = np.array([[5.0, 5.0, 5.0]])
        f_min = np.array([5.0, 5.0, 5.0])
        f_max = np.array([5.0, 5.0, 5.0])
        
        result = normalize_objectives(points, f_min, f_max)
        
        # 应该不会出现 inf 或 nan
        assert np.all(np.isfinite(result))


class TestGetHvRefPoint:
    """HV 参考点计算测试"""
    
    def test_basic_ref_point(self):
        """基本参考点计算"""
        pf_ref_norm = np.array([[0.0, 0.5, 0.3], [0.5, 0.0, 0.3], [0.3, 0.3, 0.0]])
        
        result = get_hv_ref_point(pf_ref_norm, margin=0.1)
        
        expected = np.array([0.5, 0.5, 0.3]) * 1.1
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_empty_input(self):
        """空输入"""
        pf_ref_norm = np.array([]).reshape(0, 3)
        
        result = get_hv_ref_point(pf_ref_norm)
        
        np.testing.assert_array_equal(result, [1.1, 1.1, 1.1])


class TestComputeIGD:
    """IGD 计算测试"""
    
    def test_identical_sets(self):
        """相同解集 IGD 应接近 0"""
        pf_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        A = pf_ref.copy()
        
        result = compute_igd(A, pf_ref)
        
        assert result < 0.01
    
    def test_empty_solution_set(self):
        """空解集"""
        pf_ref = np.array([[0.0, 0.0, 0.0]])
        A = np.array([]).reshape(0, 3)
        
        result = compute_igd(A, pf_ref)
        
        assert result == float('inf')


class TestComputeGD:
    """GD 计算测试"""
    
    def test_identical_sets(self):
        """相同解集 GD 应接近 0"""
        pf_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        A = pf_ref.copy()
        
        result = compute_gd(A, pf_ref)
        
        assert result < 0.01
    
    def test_distant_solution(self):
        """远离参考前沿的解"""
        pf_ref = np.array([[0.0, 0.0, 0.0]])
        A = np.array([[1.0, 1.0, 1.0]])
        
        result = compute_gd(A, pf_ref)
        
        assert result > 0


class TestComputeHV:
    """HV 计算测试"""
    
    def test_basic_hv(self):
        """基本 HV 计算"""
        A = np.array([[0.5, 0.5, 0.5]])
        ref_point = np.array([1.0, 1.0, 1.0])
        
        result = compute_hv(A, ref_point)
        
        # HV 应该是 (1-0.5)^3 = 0.125
        assert abs(result - 0.125) < 0.01
    
    def test_empty_solution_set(self):
        """空解集"""
        A = np.array([]).reshape(0, 3)
        ref_point = np.array([1.0, 1.0, 1.0])
        
        result = compute_hv(A, ref_point)
        
        assert result == 0.0
    
    def test_points_beyond_ref_point(self):
        """超出参考点的点被过滤"""
        A = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])  # 第二个点超出
        ref_point = np.array([1.0, 1.0, 1.0])
        
        result = compute_hv(A, ref_point)
        
        # 只有第一个点有效，HV 应该是 0.125
        assert abs(result - 0.125) < 0.01


class TestComputeAllMetrics:
    """综合指标计算测试"""
    
    def test_all_metrics_returned(self):
        """返回所有三个指标"""
        A = np.array([[50.0, 100.0, 150.0], [60.0, 90.0, 140.0]])
        pf_ref = np.array([[50.0, 90.0, 140.0], [60.0, 100.0, 145.0]])
        f_min = np.array([50.0, 90.0, 140.0])
        f_max = np.array([100.0, 150.0, 200.0])
        hv_ref_point = np.array([1.1, 1.1, 1.1])
        
        result = compute_all_metrics(A, pf_ref, f_min, f_max, hv_ref_point)
        
        assert 'igd' in result
        assert 'gd' in result
        assert 'hv' in result
        assert np.isfinite(result['igd'])
        assert np.isfinite(result['gd'])
        assert np.isfinite(result['hv'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
