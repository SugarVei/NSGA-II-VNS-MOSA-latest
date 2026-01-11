# -*- coding: utf-8 -*-
"""
Diagnose HV Zero Issue - Standalone Version
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder

# Import algorithms
from algorithms.nsga2 import NSGAII
from algorithms.mosa import MOSA
from algorithms.moead import MOEAD
from algorithms.spea2 import SPEA2
from algorithms.mopso import MOPSO


def is_dominated(a, b):
    """Check if a is dominated by b"""
    return all(b <= a) and any(b < a)


def get_pareto_front(objectives):
    """Extract non-dominated solutions"""
    n = len(objectives)
    is_dom = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if is_dom[i]:
            continue
        for j in range(n):
            if i != j and not is_dom[j] and is_dominated(objectives[i], objectives[j]):
                is_dom[i] = True
                break
    
    return objectives[~is_dom]


def normalize_objectives(points, f_min, f_max):
    """Min-Max normalization"""
    if len(points) == 0:
        return points.copy()
    
    range_vals = f_max - f_min
    range_vals = np.where(range_vals < 1e-12, 1e-12, range_vals)
    normalized = (points - f_min) / range_vals
    return np.clip(normalized, 0.0, None)


def compute_hv_simple(A_norm, ref_point):
    """Check how many solutions are within reference point"""
    if len(A_norm) == 0:
        return 0, 0, np.array([])
    
    valid_mask = np.all(A_norm < ref_point, axis=1)
    n_valid = np.sum(valid_mask)
    return n_valid, len(A_norm), ~valid_mask


def run_diagnostic(n_jobs=20, machines_per_stage=[3, 2, 2], seed=42):
    """Run diagnostic to compare algorithm solution quality"""
    print("=" * 70)
    print("HV Zero Issue Diagnosis")
    print("=" * 70)
    
    # Create problem instance
    print(f"\n[1] Creating test problem: {n_jobs} jobs, machines={machines_per_stage}")
    problem = SchedulingProblem.generate_random(
        n_jobs=n_jobs,
        n_stages=3,
        machines_per_stage=machines_per_stage,
        n_speed_levels=3,
        n_skill_levels=3,
        seed=seed
    )
    
    # Run algorithms (reduced iterations for faster testing)
    algorithms = {
        'NSGA-II': lambda: NSGAII(problem, pop_size=50, n_generations=50, seed=seed).run(),
        'SPEA2': lambda: SPEA2(problem, pop_size=50, archive_size=25, n_generations=50, seed=seed).run(),
        'MOPSO': lambda: MOPSO(problem, swarm_size=50, max_iterations=50, seed=seed).run(),
        'MOEA/D': lambda: MOEAD(problem, pop_size=50, n_generations=50, seed=seed).run(),
        'MOSA': lambda: MOSA(problem, initial_temp=100.0, max_iterations=100, seed=seed).run(),
    }
    
    results = {}
    print(f"\n[2] Running {len(algorithms)} algorithms...")
    
    for alg_name, run_func in algorithms.items():
        print(f"    Running {alg_name}...", end=" ", flush=True)
        try:
            pf = run_func()
            objectives = np.array([s.objectives for s in pf if s.objectives is not None])
            results[alg_name] = objectives
            print(f"Done, got {len(objectives)} solutions")
        except Exception as e:
            print(f"Error: {e}")
            results[alg_name] = np.array([]).reshape(0, 3)
    
    # Build global reference front
    print(f"\n[3] Building global reference front...")
    all_objectives = np.vstack([obj for obj in results.values() if len(obj) > 0])
    pf_ref = get_pareto_front(all_objectives)
    print(f"    Merged {len(all_objectives)} solutions")
    print(f"    Reference front has {len(pf_ref)} non-dominated solutions")
    
    # Normalization parameters
    f_min = np.min(pf_ref, axis=0)
    f_max = np.max(pf_ref, axis=0)
    hv_ref_point = np.array([1.1, 1.1, 1.1])
    
    print(f"\n[4] Normalization parameters (from reference front):")
    print(f"    f_min = [{f_min[0]:.2f}, {f_min[1]:.2f}, {f_min[2]:.2f}]")
    print(f"    f_max = [{f_max[0]:.2f}, {f_max[1]:.2f}, {f_max[2]:.2f}]")
    print(f"    HV reference point (normalized) = {hv_ref_point}")
    
    # Analyze raw objective ranges
    print(f"\n[5] RAW objective value ranges per algorithm:")
    print("-" * 80)
    print(f"{'Algorithm':<12} {'F1 (Makespan)':<22} {'F2 (Labor Cost)':<22} {'F3 (Energy)':<22}")
    print("-" * 80)
    
    for alg_name, objectives in results.items():
        if len(objectives) > 0:
            f1_r = f"{objectives[:, 0].min():.1f} ~ {objectives[:, 0].max():.1f}"
            f2_r = f"{objectives[:, 1].min():.1f} ~ {objectives[:, 1].max():.1f}"
            f3_r = f"{objectives[:, 2].min():.1f} ~ {objectives[:, 2].max():.1f}"
            print(f"{alg_name:<12} {f1_r:<22} {f2_r:<22} {f3_r:<22}")
        else:
            print(f"{alg_name:<12} (no valid solutions)")
    print("-" * 80)
    print(f"\n    Reference front range: F1={f_min[0]:.1f}~{f_max[0]:.1f}, F2={f_min[1]:.1f}~{f_max[1]:.1f}, F3={f_min[2]:.1f}~{f_max[2]:.1f}")
    
    # Analyze normalized ranges
    print(f"\n[6] NORMALIZED objective ranges (ref front [0, 1]):")
    print("-" * 80)
    print(f"{'Algorithm':<12} {'F1_norm range':<22} {'F2_norm range':<22} {'F3_norm range':<22}")
    print("-" * 80)
    
    for alg_name, objectives in results.items():
        if len(objectives) > 0:
            A_norm = normalize_objectives(objectives, f_min, f_max)
            f1_r = f"{A_norm[:, 0].min():.3f} ~ {A_norm[:, 0].max():.3f}"
            f2_r = f"{A_norm[:, 1].min():.3f} ~ {A_norm[:, 1].max():.3f}"
            f3_r = f"{A_norm[:, 2].min():.3f} ~ {A_norm[:, 2].max():.3f}"
            print(f"{alg_name:<12} {f1_r:<22} {f2_r:<22} {f3_r:<22}")
        else:
            print(f"{alg_name:<12} (no valid solutions)")
    print("-" * 80)
    
    # Analyze HV validity
    print(f"\n[7] HV validity analysis (ref_point = {hv_ref_point}):")
    print("-" * 80)
    print(f"{'Algorithm':<12} {'Total':<10} {'Valid':<10} {'Exceed%':<12} {'Status':<20}")
    print("-" * 80)
    
    for alg_name, objectives in results.items():
        if len(objectives) > 0:
            A_norm = normalize_objectives(objectives, f_min, f_max)
            n_valid, total, exceed_mask = compute_hv_simple(A_norm, hv_ref_point)
            exceed_ratio = (total - n_valid) / total * 100
            
            status = "OK" if n_valid > 0 else "HV=0 (all exceed)"
            print(f"{alg_name:<12} {total:<10} {n_valid:<10} {exceed_ratio:.1f}%{'':<6} {status:<20}")
            
            # Show details of exceeding solutions
            if n_valid < total:
                exceed_solutions = A_norm[exceed_mask]
                print(f"    WARNING: {len(exceed_solutions)} solutions exceed reference point:")
                for idx, sol in enumerate(exceed_solutions[:3]):
                    exceed_dims = []
                    for d in range(3):
                        if sol[d] >= hv_ref_point[d]:
                            exceed_dims.append(f"F{d+1}={sol[d]:.3f} >= 1.1")
                    print(f"        Sol {idx+1}: {', '.join(exceed_dims)}")
                if len(exceed_solutions) > 3:
                    print(f"        ... and {len(exceed_solutions) - 3} more")
        else:
            print(f"{alg_name:<12} 0          0          N/A          NO SOLUTIONS")
    print("-" * 80)
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
ROOT CAUSE of HV = 0:
1. Normalization is based on the MERGED reference front from ALL algorithms
2. If an algorithm produces solutions with objectives far exceeding the
   reference front's range, after normalization they exceed [0, 1]
3. HV computation filters out solutions with ANY dimension >= ref_point (1.1)
4. If ALL solutions are filtered out, HV = 0

SOLUTIONS:
1. Increase iterations for underperforming algorithms (MOSA, MOEA/D)
2. Tune algorithm parameters to improve solution quality
3. Increase HV reference point margin (but affects comparability)
""")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
