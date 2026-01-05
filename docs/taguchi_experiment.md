# 田口实验调参模块使用说明

## 概述

本模块为 HYBRID 算法（NSGA-II + VNS + MOSA）提供**田口实验法**参数调优功能，使用 **pymoo** 计算 IGD/GD/HV 指标，输出论文友好的统计表与图。

## 安装依赖

```bash
cd scheduling_optimizer
pip install -r requirements.txt
```

新增依赖：
- `pymoo>=0.6.0` - 多目标优化指标计算
- `tqdm>=4.60.0` - 进度条
- `pyyaml>=6.0` - 配置文件
- `joblib>=1.2.0` - 并行计算（预留）

## 快速开始

### 运行完整田口实验（默认30次重复）

```bash
cd scheduling_optimizer
python -m experiments.taguchi.run_taguchi --rep 30 --out results/taguchi
```

### 小规模测试（2次重复）

```bash
python -m experiments.taguchi.run_taguchi --rep 2 --out results/taguchi_test
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--case` | `real_case_15x3` | 问题实例名称 |
| `--rep` | `30` | 每组重复次数 |
| `--base-seed` | `20250101` | 基础随机种子 |
| `--out` | `results/taguchi` | 输出目录 |
| `--n-jobs` | `1` | 并行任务数 |
| `--skip-confirm` | - | 跳过确认实验 |
| `--quiet` | - | 安静模式 |

## 实验设计

### 因子与水平

| 因子 | 参数名 | 水平1 | 水平2 | 水平3 | 水平4 |
|------|--------|-------|-------|-------|-------|
| A | population_size | 50 | 100 | 150 | 200 |
| B | crossover_prob | 0.70 | 0.80 | 0.90 | 0.95 |
| C | mutation_prob | 0.05 | 0.10 | 0.15 | 0.20 |
| D | initial_temp | 100 | 300 | 500 | 1000 |

### L16(4^4) 正交表

共 16 组实验配置，每组重复 30 次（可配置），总计 480 次算法运行。

### 固定问题实例

- 工件数：15
- 阶段数：3
- 每阶段平行机：3
- 工人技能人数：A=7, B=5, C=3

## 输出文件

运行完成后，输出目录结构如下：

```
results/taguchi/
├── taguchi_runs.csv           # 所有运行的详细记录
├── taguchi_summary.csv        # 16组汇总统计
├── pf_ref.csv                 # 全局参考前沿
├── normalization.json         # 归一化参数
├── hv_ref_point.json          # HV 参考点
├── snr_effect_igd.csv         # IGD 主效应表
├── snr_effect_hv.csv          # HV 主效应表
├── snr_effect_gd.csv          # GD 主效应表
├── best_params.json           # 推荐最优参数
├── confirm_experiment_summary.csv  # 确认实验结果
├── error_log.txt              # 异常记录
└── figures/
    ├── main_effect_igd.png    # IGD 主效应图
    ├── main_effect_hv.png     # HV 主效应图
    ├── main_effect_gd.png     # GD 主效应图
    ├── boxplot_igd.png        # IGD 分布箱线图
    ├── boxplot_hv.png         # HV 分布箱线图
    ├── boxplot_gd.png         # GD 分布箱线图
    ├── pareto_f1_f2.png       # Pareto 投影 (F1,F2)
    ├── pareto_f1_f3.png       # Pareto 投影 (F1,F3)
    ├── pareto_f2_f3.png       # Pareto 投影 (F2,F3)
    ├── pareto_3d.png          # 3D Pareto 图
    └── confirmation_comparison.png  # 确认实验对比图
```

## 论文写法参考

### PF_ref 构造

> 由于真实调度问题无法获得理论 Pareto 前沿，本文采用"全局参考前沿"（PF_ref）构造策略。对同一固定问题实例，将田口实验 L16 正交表设计下 16×30=480 次运行得到的所有非支配解目标点合并为全集 U，在 U 上执行全局非支配筛选，得到近似真实 Pareto 前沿 PF_ref。

### 指标归一化

> 所有指标（IGD、GD、HV）均在 Min-Max 归一化空间中计算。归一化参数（f_min, f_max）取自 PF_ref 的各目标极值。归一化公式为：
> 
> f_norm = (f - f_min) / (f_max - f_min)

### HV 参考点

> HV 的参考点设置为归一化 PF_ref 各目标最大值的 1.1 倍（即 110%），确保所有非支配解均落在参考点支配的区域内。

### S/N 比计算

- **Smaller-the-better**（IGD、GD）：SNR = -10 × log₁₀(mean(y²))
- **Larger-the-better**（HV）：SNR = -10 × log₁₀(mean(1/y²))

## 单元测试

```bash
cd scheduling_optimizer
python -m pytest tests/test_pareto.py tests/test_metrics.py -v
```

## 模块结构

```
experiments/taguchi/
├── __init__.py         # 模块导出
├── designs.py          # L16 正交表、因子水平
├── pareto.py           # 非支配筛选、PF_ref 构造
├── metrics.py          # IGD/GD/HV (pymoo)
├── analysis.py         # SNR、主效应分析
├── plotting.py         # 可视化
├── io.py               # 文件 I/O
└── run_taguchi.py      # CLI 入口
```
