# -*- coding: utf-8 -*-
"""
田口实验可视化窗口
Taguchi Experiment Visualization Window

提供田口实验的配置、运行和结果展示界面。
"""

import sys
import os
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QScrollArea, QFrame, QSplitter,
    QTextEdit, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.problem import SchedulingProblem


# ==================== 样式常量 ====================
TAGUCHI_STYLE = """
QMainWindow, QWidget {
    background-color: #E3F2FD;
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
}
QGroupBox {
    font-weight: bold;
    font-size: 14px;
    border: 2px solid #64B5F6;
    border-radius: 8px;
    margin-top: 12px;
    padding: 15px 10px 10px 10px;
    background-color: #BBDEFB;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 2px 8px;
    color: #1565C0;
    background-color: #90CAF9;
    border-radius: 4px;
}
QPushButton {
    background-color: #1976D2;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1565C0;
}
QPushButton:disabled {
    background-color: #90CAF9;
    color: #BBDEFB;
}
QLabel {
    color: #0D47A1;
    font-size: 13px;
    background-color: transparent;
}
QProgressBar {
    border: 2px solid #64B5F6;
    border-radius: 5px;
    text-align: center;
    background-color: white;
}
QProgressBar::chunk {
    background-color: #4CAF50;
}
QTableWidget {
    background-color: white;
    gridline-color: #90CAF9;
    border-radius: 4px;
}
QTabWidget::pane {
    border: 1px solid #64B5F6;
    border-radius: 4px;
    background-color: white;
}
"""


class TaguchiWorker(QThread):
    """田口实验后台运行线程"""
    
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # 结果字典
    error = pyqtSignal(str)  # 错误消息
    
    def __init__(self, n_rep: int, base_seed: int, output_dir: str):
        super().__init__()
        self.n_rep = n_rep
        self.base_seed = base_seed
        self.output_dir = output_dir
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        try:
            from experiments.taguchi.run_taguchi import (
                create_problem_instance, run_taguchi_experiment,
                build_pf_ref, compute_metrics_for_runs,
                compute_all_snr_effects, recommend_best_params,
                create_all_figures
            )
            from experiments.taguchi.pareto import build_pf_ref
            from experiments.taguchi.metrics import get_normalization_info
            from experiments.taguchi.io import (
                save_csv, save_json, save_pf_ref, save_normalization,
                save_hv_ref_point, create_output_structure, ErrorLogger
            )
            
            # 创建输出目录
            paths = create_output_structure(Path(self.output_dir))
            error_logger = ErrorLogger(paths['error_log'])
            
            # 创建问题实例
            self.progress.emit(0, 100, "正在创建问题实例...")
            problem = create_problem_instance()
            
            from experiments.taguchi.designs import REAL_CASE_PARAMS
            problem.workers_available = REAL_CASE_PARAMS['workers_available']
            
            # 运行田口实验
            self.progress.emit(5, 100, "正在运行田口实验...")
            
            # 添加进度回调
            runs_df, all_objectives = self._run_experiment_with_progress(
                problem, self.n_rep, self.base_seed, error_logger
            )
            
            if not self._is_running:
                return
            
            # 构造 PF_ref
            self.progress.emit(85, 100, "正在构造参考前沿...")
            pf_ref = build_pf_ref(all_objectives)
            save_pf_ref(pf_ref, paths['pf_ref'])
            
            # 计算归一化参数
            norm_info = get_normalization_info(pf_ref)
            f_min = np.array(norm_info['f_min'])
            f_max = np.array(norm_info['f_max'])
            hv_ref_point = np.array(norm_info['hv_ref_point'])
            
            save_normalization(f_min, f_max, paths['normalization'])
            save_hv_ref_point(hv_ref_point, paths['hv_ref_point'])
            
            # 计算指标
            self.progress.emit(90, 100, "正在计算指标...")
            runs_df = compute_metrics_for_runs(
                runs_df, all_objectives, pf_ref, f_min, f_max, hv_ref_point
            )
            save_csv(runs_df, paths['taguchi_runs'])
            
            # 计算主效应
            self.progress.emit(92, 100, "正在分析主效应...")
            effects = compute_all_snr_effects(runs_df)
            save_csv(effects['igd'], paths['snr_effect_igd'], index=True)
            save_csv(effects['hv'], paths['snr_effect_hv'], index=True)
            save_csv(effects['gd'], paths['snr_effect_gd'], index=True)
            
            # 推荐最优参数
            best_result = recommend_best_params(effects['igd'], effects['hv'], effects['gd'])
            save_json(best_result, paths['best_params'])
            
            # 生成图表
            self.progress.emit(95, 100, "正在生成图表...")
            
            # 取一个代表性解集
            best_run_objectives = np.array([]).reshape(0, 3)
            for obj in all_objectives:
                if len(obj) > 0:
                    best_run_objectives = obj
                    break
            
            create_all_figures(
                df=runs_df,
                effects=effects,
                pf_ref=pf_ref,
                best_run_objectives=best_run_objectives,
                figures_dir=paths['figures'],
                confirm_df=None,
                default_df=None
            )
            
            error_logger.finalize()
            
            self.progress.emit(100, 100, "完成!")
            
            # 返回结果
            result = {
                'success': True,
                'output_dir': str(paths['base']),
                'figures_dir': str(paths['figures']),
                'best_params': best_result['best_params'],
                'best_levels': best_result['best_levels'],
                'runs_df': runs_df,
                'effects': effects,
                'pf_ref_size': len(pf_ref),
                'error_count': error_logger.get_error_count(),
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _run_experiment_with_progress(self, problem, n_rep, base_seed, error_logger):
        """运行实验并更新进度"""
        from experiments.taguchi.designs import get_params_for_run, get_level_indices_for_run, get_n_runs
        from experiments.taguchi.run_taguchi import run_hybrid
        
        n_runs = get_n_runs()
        total_runs = n_runs * n_rep
        
        results = []
        all_objectives = []
        
        for run_id in range(n_runs):
            if not self._is_running:
                break
                
            params = get_params_for_run(run_id)
            level_indices = get_level_indices_for_run(run_id)
            
            for rep_id in range(n_rep):
                if not self._is_running:
                    break
                
                current = run_id * n_rep + rep_id + 1
                progress_pct = 5 + int(80 * current / total_runs)
                self.progress.emit(progress_pct, 100, 
                                   f"运行 {run_id+1}/16, 重复 {rep_id+1}/{n_rep}")
                
                seed = base_seed + run_id * 1000 + rep_id
                
                try:
                    objectives, time_sec = run_hybrid(problem, params, seed)
                    n_solutions = len(objectives)
                    
                    if len(objectives) > 0:
                        all_objectives.append(objectives)
                    
                    result = {
                        'run_id': run_id,
                        'rep_id': rep_id,
                        **level_indices,
                        'population_size': params['population_size'],
                        'crossover_prob': params['crossover_prob'],
                        'mutation_prob': params['mutation_prob'],
                        'initial_temp': params['initial_temp'],
                        'seed': seed,
                        'time_sec': time_sec,
                        'n_solutions': n_solutions,
                        'igd': np.nan,
                        'gd': np.nan,
                        'hv': np.nan,
                    }
                    
                except Exception as e:
                    error_logger.log_error(run_id, rep_id, params, seed, e)
                    result = {
                        'run_id': run_id,
                        'rep_id': rep_id,
                        **level_indices,
                        'population_size': params['population_size'],
                        'crossover_prob': params['crossover_prob'],
                        'mutation_prob': params['mutation_prob'],
                        'initial_temp': params['initial_temp'],
                        'seed': seed,
                        'time_sec': np.nan,
                        'n_solutions': 0,
                        'igd': np.nan,
                        'gd': np.nan,
                        'hv': np.nan,
                    }
                    all_objectives.append(np.array([]).reshape(0, 3))
                
                results.append(result)
        
        return pd.DataFrame(results), all_objectives


class TaguchiWindow(QMainWindow):
    """田口实验可视化窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.result = None
        self.setup_ui()
        self.setStyleSheet(TAGUCHI_STYLE)
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle("🔬 田口实验 - 参数调优")
        self.setMinimumSize(1000, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ===== 标题 =====
        title = QLabel("🔬 田口实验 (Taguchi Design) - L16(4⁴) 正交表参数调优")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; padding: 10px; background-color: #1565C0; border-radius: 6px;")
        main_layout.addWidget(title)
        
        # ===== 内容区域 =====
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：配置面板
        left_panel = self.create_config_panel()
        content_splitter.addWidget(left_panel)
        
        # 右侧：结果标签页
        self.result_tabs = self.create_result_tabs()
        content_splitter.addWidget(self.result_tabs)
        
        content_splitter.setSizes([350, 650])
        main_layout.addWidget(content_splitter)
        
        # ===== 底部按钮 =====
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("🚀 开始实验")
        self.run_btn.clicked.connect(self.on_start_experiment)
        btn_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_experiment)
        self.stop_btn.setStyleSheet("background-color: #f44336;")
        btn_layout.addWidget(self.stop_btn)
        
        btn_layout.addStretch()
        
        self.export_btn = QPushButton("📁 打开结果目录")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_open_results)
        btn_layout.addWidget(self.export_btn)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("background-color: #757575;")
        btn_layout.addWidget(close_btn)
        
        main_layout.addLayout(btn_layout)
    
    def create_config_panel(self) -> QWidget:
        """创建配置面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # ===== 实验设置 =====
        settings_group = QGroupBox("⚙️ 实验设置")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("重复次数:"), 0, 0)
        self.rep_spin = QSpinBox()
        self.rep_spin.setRange(1, 100)
        self.rep_spin.setValue(5)
        self.rep_spin.setToolTip("每组实验重复次数（完整实验建议30次）")
        settings_layout.addWidget(self.rep_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("随机种子:"), 1, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999999)
        self.seed_spin.setValue(20250101)
        settings_layout.addWidget(self.seed_spin, 1, 1)
        
        layout.addWidget(settings_group)
        
        # ===== 因子水平表 =====
        factors_group = QGroupBox("📊 因子水平 (4因子×4水平)")
        factors_layout = QVBoxLayout(factors_group)
        
        factors_table = QTableWidget()
        factors_table.setRowCount(4)
        factors_table.setColumnCount(5)
        factors_table.setHorizontalHeaderLabels(["因子", "水平1", "水平2", "水平3", "水平4"])
        factors_table.setVerticalHeaderLabels(["A", "B", "C", "D"])
        
        factors_data = [
            ("population_size", "50", "100", "150", "200"),
            ("crossover_prob", "0.70", "0.80", "0.90", "0.95"),
            ("mutation_prob", "0.05", "0.10", "0.15", "0.20"),
            ("initial_temp", "100", "300", "500", "1000"),
        ]
        
        for row, (name, *levels) in enumerate(factors_data):
            factors_table.setItem(row, 0, QTableWidgetItem(name))
            for col, level in enumerate(levels):
                factors_table.setItem(row, col + 1, QTableWidgetItem(level))
        
        factors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        factors_table.setEditTriggers(QTableWidget.NoEditTriggers)
        factors_table.setMaximumHeight(150)
        factors_layout.addWidget(factors_table)
        
        layout.addWidget(factors_group)
        
        # ===== 问题实例 =====
        problem_group = QGroupBox("📦 固定问题实例")
        problem_layout = QVBoxLayout(problem_group)
        
        problem_info = QLabel(
            "• 工件数: 15\n"
            "• 阶段数: 3\n"
            "• 每阶段机器: 3\n"
            "• 工人技能人数: A=7, B=5, C=3"
        )
        problem_info.setStyleSheet("font-size: 12px; padding: 5px;")
        problem_layout.addWidget(problem_info)
        
        layout.addWidget(problem_group)
        
        # ===== 进度 =====
        progress_group = QGroupBox("📈 运行进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("等待开始...")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
        return widget
    
    def create_result_tabs(self) -> QTabWidget:
        """创建结果标签页"""
        tabs = QTabWidget()
        
        # 推荐参数标签页
        self.params_tab = self.create_params_tab()
        tabs.addTab(self.params_tab, "📋 推荐参数")
        
        # 主效应图标签页
        self.effects_tab = self.create_image_tab("主效应图将在实验完成后显示")
        tabs.addTab(self.effects_tab, "📊 主效应图")
        
        # 箱线图标签页
        self.boxplot_tab = self.create_image_tab("箱线图将在实验完成后显示")
        tabs.addTab(self.boxplot_tab, "📦 箱线图")
        
        # Pareto图标签页
        self.pareto_tab = self.create_image_tab("Pareto图将在实验完成后显示")
        tabs.addTab(self.pareto_tab, "🎯 Pareto图")
        
        return tabs
    
    def create_params_tab(self) -> QWidget:
        """创建推荐参数标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.params_label = QLabel("实验完成后将在此显示推荐的最优参数组合")
        self.params_label.setAlignment(Qt.AlignCenter)
        self.params_label.setFont(QFont("Microsoft YaHei", 12))
        self.params_label.setStyleSheet("color: #666; padding: 50px;")
        layout.addWidget(self.params_label)
        
        return widget
    
    def create_image_tab(self, placeholder_text: str) -> QWidget:
        """创建图片显示标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        placeholder = QLabel(placeholder_text)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-size: 14px; padding: 50px;")
        content_layout.addWidget(placeholder)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return widget
    
    def on_start_experiment(self):
        """开始实验"""
        n_rep = self.rep_spin.value()
        base_seed = self.seed_spin.value()
        
        # 确定输出目录
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "taguchi_ui"
        )
        
        # 确认
        total_runs = 16 * n_rep
        est_time = total_runs * 10 / 60  # 估计每次运行10秒
        
        reply = QMessageBox.question(
            self, "确认开始",
            f"将运行 16×{n_rep}={total_runs} 次实验\n"
            f"预计耗时: {est_time:.1f} 分钟\n\n"
            f"是否开始?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 禁用开始按钮
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.rep_spin.setEnabled(False)
        self.seed_spin.setEnabled(False)
        
        # 启动工作线程
        self.worker = TaguchiWorker(n_rep, base_seed, output_dir)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def on_stop_experiment(self):
        """停止实验"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
            self.status_label.setText("已停止")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.rep_spin.setEnabled(True)
            self.seed_spin.setEnabled(True)
    
    def on_progress(self, current: int, total: int, message: str):
        """更新进度"""
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def on_finished(self, result: dict):
        """实验完成"""
        self.result = result
        
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        self.rep_spin.setEnabled(True)
        self.seed_spin.setEnabled(True)
        
        self.status_label.setText("✅ 实验完成!")
        
        # 更新推荐参数
        best_params = result['best_params']
        best_levels = result['best_levels']
        
        params_text = f"""
<h2>🎯 推荐最优参数组合</h2>
<table border="1" cellpadding="8" style="border-collapse: collapse; margin: 10px;">
<tr><th>因子</th><th>水平</th><th>参数名</th><th>值</th></tr>
<tr><td>A</td><td>{best_levels['A']}</td><td>population_size</td><td><b>{best_params['population_size']}</b></td></tr>
<tr><td>B</td><td>{best_levels['B']}</td><td>crossover_prob</td><td><b>{best_params['crossover_prob']}</b></td></tr>
<tr><td>C</td><td>{best_levels['C']}</td><td>mutation_prob</td><td><b>{best_params['mutation_prob']}</b></td></tr>
<tr><td>D</td><td>{best_levels['D']}</td><td>initial_temp</td><td><b>{best_params['initial_temp']}</b></td></tr>
</table>
<p>PF_ref 大小: {result['pf_ref_size']} 个非支配解</p>
<p>错误数量: {result['error_count']}</p>
"""
        self.params_label.setText(params_text)
        self.params_label.setTextFormat(Qt.RichText)
        self.params_label.setStyleSheet("padding: 20px; background-color: white;")
        
        # 加载图片
        figures_dir = Path(result['figures_dir'])
        
        self._load_images_to_tab(self.effects_tab, [
            figures_dir / 'main_effect_igd.png',
            figures_dir / 'main_effect_hv.png',
            figures_dir / 'main_effect_gd.png',
        ])
        
        self._load_images_to_tab(self.boxplot_tab, [
            figures_dir / 'boxplot_igd.png',
            figures_dir / 'boxplot_hv.png',
            figures_dir / 'boxplot_gd.png',
        ])
        
        self._load_images_to_tab(self.pareto_tab, [
            figures_dir / 'pareto_f1_f2.png',
            figures_dir / 'pareto_f1_f3.png',
            figures_dir / 'pareto_f2_f3.png',
        ])
        
        QMessageBox.information(self, "完成", 
                                f"田口实验完成!\n\n"
                                f"推荐参数:\n"
                                f"  population_size: {best_params['population_size']}\n"
                                f"  crossover_prob: {best_params['crossover_prob']}\n"
                                f"  mutation_prob: {best_params['mutation_prob']}\n"
                                f"  initial_temp: {best_params['initial_temp']}\n\n"
                                f"结果已保存到: {result['output_dir']}")
    
    def _load_images_to_tab(self, tab: QWidget, image_paths: list):
        """加载图片到标签页"""
        # 清除现有内容
        layout = tab.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        for path in image_paths:
            if path.exists():
                img_label = QLabel()
                pixmap = QPixmap(str(path))
                # 缩放到合适大小
                scaled = pixmap.scaledToWidth(600, Qt.SmoothTransformation)
                img_label.setPixmap(scaled)
                img_label.setAlignment(Qt.AlignCenter)
                content_layout.addWidget(img_label)
                
                # 添加文件名标签
                name_label = QLabel(path.name)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 15px;")
                content_layout.addWidget(name_label)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
    
    def on_error(self, error_msg: str):
        """处理错误"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.rep_spin.setEnabled(True)
        self.seed_spin.setEnabled(True)
        
        self.status_label.setText("❌ 发生错误")
        QMessageBox.critical(self, "错误", f"实验运行出错:\n\n{error_msg}")
    
    def on_open_results(self):
        """打开结果目录"""
        if self.result and 'output_dir' in self.result:
            import subprocess
            subprocess.Popen(f'explorer "{self.result["output_dir"]}"')
    
    def closeEvent(self, event):
        """关闭窗口时停止工作线程"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = TaguchiWindow()
    window.show()
    sys.exit(app.exec_())
