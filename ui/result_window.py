"""
结果展示窗口
Result Window

显示优化结果:
- 双进度条 (NSGA-II / VNS+MOSA)
- 收敛曲线
- 3D Pareto 前沿散点图
- 甘特图
- 解列表 (按综合值排序)
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QTabWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QFileDialog, QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush

import numpy as np
from typing import List, Dict

# Matplotlib
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from algorithms.nsga2 import NSGAII
from algorithms.mosa import MOSA


# 样式
RESULT_STYLE = """
QMainWindow {
    background-color: #f8f9fa;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    background-color: white;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    color: #495057;
}
QPushButton {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0056b3;
}
QPushButton:disabled {
    background-color: #6c757d;
}
QProgressBar {
    border: 1px solid #dee2e6;
    border-radius: 4px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #28a745;
}
QTableWidget {
    gridline-color: #dee2e6;
    selection-background-color: #007bff;
}
"""


class OptimizationWorker(QThread):
    """优化算法工作线程"""
    
    # 双进度信号
    nsga2_progress = pyqtSignal(int, int, str)  # current, total, message
    mosa_progress = pyqtSignal(int, int, str)
    
    log = pyqtSignal(str)
    nsga2_finished = pyqtSignal(list, dict)
    mosa_finished = pyqtSignal(list, dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, problem: SchedulingProblem, params: dict):
        super().__init__()
        self.problem = problem
        self.params = params
        self._is_cancelled = False
    
    def run(self):
        """运行优化"""
        try:
            from algorithms.vns import VNS
            
            # 1. NSGA-II
            self.log.emit(f"[{self._time()}] Starting NSGA-II...")
            self.nsga2_progress.emit(0, 100, "Initializing...")
            
            nsga2 = NSGAII(
                problem=self.problem,
                pop_size=self.params['pop_size'],
                n_generations=self.params['n_generations'],
                crossover_prob=self.params['crossover_prob'],
                mutation_prob=self.params['mutation_prob'],
                seed=self.params.get('seed', 42)
            )
            
            def nsga2_callback(cur, total, msg):
                if self._is_cancelled: return
                prog = int(cur / total * 100)
                self.nsga2_progress.emit(prog, 100, msg)
                if cur % 20 == 0:
                    self.log.emit(f"  {msg}")
            
            nsga2.set_progress_callback(nsga2_callback)
            pareto_nsga2 = nsga2.run()
            
            self.log.emit(f"[{self._time()}] NSGA-II 完成，找到 {len(pareto_nsga2)} 个 Pareto 解")
            self.nsga2_finished.emit(pareto_nsga2, nsga2.get_convergence_data())
            
            if self._is_cancelled: return
            
            # 2. VNS + MOSA 分开迭代
            n_pareto = len(pareto_nsga2)
            self.log.emit(f"[{self._time()}] 开始 VNS-MOSA 迭代优化 (共 {n_pareto} 个解)...")
            self.log.emit("=" * 50)
            
            # 初始化VNS (带SA参数)
            from models.decoder import Decoder
            decoder = Decoder(self.problem)
            vns = VNS(
                problem=self.problem,
                max_iters=10,  # VNS-MOSA内部迭代
                seed=self.params.get('seed', 42)
            )

            
            # MOSA 参数
            temperature = self.params['initial_temp']
            cooling_rate = self.params['cooling_rate']
            weights = np.array(self.params['weights'])
            
            # 存储优化后的解
            archive = list(pareto_nsga2)
            
            # 对每个Pareto解进行VNS+MOSA
            for i, current_solution in enumerate(pareto_nsga2):
                if self._is_cancelled: return
                
                prog = int((i + 1) / n_pareto * 100)
                
                # VNS 邻域搜索
                self.log.emit(f"[迭代 {i+1}/{n_pareto}]")
                self.log.emit(f"  ├─ VNS: 邻域搜索...")
                self.mosa_progress.emit(prog, 100, f"VNS {i+1}/{n_pareto}")
                
                vns_solution = vns.run(current_solution)
                
                # 确保有目标值
                if vns_solution.objectives is None:
                    vns_solution.objectives = decoder.decode(vns_solution)
                
                # MOSA 模拟退火接受判断
                self.log.emit(f"  ├─ MOSA: 接受判断 (温度 T={temperature:.2f})")
                
                # 计算加权和 (不归一化)
                current_obj = np.array(current_solution.objectives)
                new_obj = np.array(vns_solution.objectives)
                
                current_score = np.dot(current_obj, weights)
                new_score = np.dot(new_obj, weights)
                
                delta = new_score - current_score
                
                # 接受条件
                if delta < 0:
                    # 新解更好，直接接受
                    accepted = True
                    reason = "新解更优"
                else:
                    # 按概率接受
                    prob = np.exp(-delta / max(temperature, 0.001))
                    accepted = np.random.random() < prob
                    reason = f"接受概率={prob:.3f}"
                
                if accepted:
                    self.log.emit(f"  └─ ✓ 接受 ({reason})")
                    archive.append(vns_solution)
                else:
                    self.log.emit(f"  └─ ✗ 拒绝 ({reason})")
                
                # 降温
                temperature *= cooling_rate
            
            self.log.emit("=" * 50)
            
            # 去除被支配的解，得到最终Pareto前沿
            self.log.emit(f"[{self._time()}] 过滤被支配的解...")
            final_archive = self._get_pareto_front(archive)
            
            self.log.emit(f"[{self._time()}] VNS+MOSA 完成，最终 Pareto 解数量: {len(final_archive)}")
            
            # 构造收敛数据
            mosa_conv_data = {
                'iteration': list(range(n_pareto)),
                'temperature': [self.params['initial_temp'] * (self.params['cooling_rate'] ** i) for i in range(n_pareto)]
            }
            
            self.mosa_finished.emit(final_archive, mosa_conv_data)
            
            self.nsga2_progress.emit(100, 100, "完成")
            self.mosa_progress.emit(100, 100, "完成")
            self.log.emit(f"[{self._time()}] ✅ 优化完成!")

            
        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    
    def _time(self):
        return datetime.now().strftime('%H:%M:%S')
    
    def cancel(self):
        self._is_cancelled = True
    
    def _get_pareto_front(self, solutions: list) -> list:
        """过滤被支配的解，返回Pareto前沿"""
        if not solutions:
            return []
        
        pareto = []
        for sol in solutions:
            if sol.objectives is None:
                continue
            
            dominated = False
            for other in solutions:
                if other.objectives is None or other is sol:
                    continue
                
                # 检查是否被支配
                all_leq = all(o <= s for o, s in zip(other.objectives, sol.objectives))
                any_lt = any(o < s for o, s in zip(other.objectives, sol.objectives))
                
                if all_leq and any_lt:
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(sol)
        
        # 去重
        unique_pareto = []
        seen = set()
        for sol in pareto:
            key = tuple(round(o, 4) for o in sol.objectives)
            if key not in seen:
                seen.add(key)
                unique_pareto.append(sol)
        
        return unique_pareto


class ResultWindow(QMainWindow):
    """结果展示窗口"""
    
    def __init__(self, problem: SchedulingProblem, params: dict):
        super().__init__()
        self.problem = problem
        self.params = params
        self.pareto_solutions = []
        self.convergence_data = {}
        self.worker = None
        self.best_solution_idx = -1
        self.setup_ui()
        self.setStyleSheet(RESULT_STYLE)
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle("优化结果")
        self.setMinimumSize(1200, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 进度区 - 双进度条
        progress_group = QGroupBox("📊 优化进度")
        progress_layout = QGridLayout(progress_group)
        
        # NSGA-II 进度
        progress_layout.addWidget(QLabel("NSGA-II:"), 0, 0)
        self.nsga2_label = QLabel("就绪")
        progress_layout.addWidget(self.nsga2_label, 0, 1)
        self.nsga2_bar = QProgressBar()
        self.nsga2_bar.setValue(0)
        progress_layout.addWidget(self.nsga2_bar, 0, 2)
        
        # VNS+MOSA 进度
        progress_layout.addWidget(QLabel("VNS+MOSA:"), 1, 0)
        self.mosa_label = QLabel("等待中...")
        progress_layout.addWidget(self.mosa_label, 1, 1)
        self.mosa_bar = QProgressBar()
        self.mosa_bar.setValue(0)
        self.mosa_bar.setStyleSheet("QProgressBar::chunk { background-color: #17a2b8; }")
        progress_layout.addWidget(self.mosa_bar, 1, 2)
        
        main_layout.addWidget(progress_group)
        
        # 主内容区
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧: 日志
        log_group = QGroupBox("📝 运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setMaximumWidth(400)
        splitter.addWidget(log_group)
        
        # 右侧: 结果标签页
        self.tabs = QTabWidget()
        
        # 收敛曲线
        self.convergence_canvas = self.create_canvas()
        self.tabs.addTab(self.convergence_canvas, "📈 收敛曲线")
        
        # 3D Pareto
        self.pareto_canvas = self.create_canvas()
        self.tabs.addTab(self.pareto_canvas, "🎯 Pareto 3D")
        
        # 2D Pareto 对比图
        self.pareto_2d_canvas = self.create_canvas()
        self.tabs.addTab(self.pareto_2d_canvas, "📊 Pareto 2D")
        
        # 甘特图 (带缩放工具栏)
        gantt_widget = QWidget()
        gantt_layout = QVBoxLayout(gantt_widget)
        gantt_layout.setContentsMargins(0, 0, 0, 0)
        
        self.gantt_canvas = self.create_canvas()
        
        # 添加导航工具栏 (支持放大、缩小、平移)
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.gantt_toolbar = NavigationToolbar(self.gantt_canvas, gantt_widget)
        self.gantt_toolbar.setStyleSheet("background-color: #f0f0f0;")
        
        gantt_layout.addWidget(self.gantt_toolbar)
        gantt_layout.addWidget(self.gantt_canvas)
        
        self.tabs.addTab(gantt_widget, "📅 甘特图")
        
        # 解列表
        self.solution_table = self.create_solution_table()
        self.tabs.addTab(self.solution_table, "📋 解列表")
        
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 850])
        
        main_layout.addWidget(splitter)
        
        # 按钮区
        btn_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("⏹ 取消")
        self.cancel_btn.clicked.connect(self.cancel_optimization)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)
        
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("💾 保存图表")
        self.save_btn.clicked.connect(self.save_figures)
        self.save_btn.setEnabled(False)
        btn_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(btn_layout)
    
    def create_canvas(self) -> FigureCanvas:
        """创建 Matplotlib 画布"""
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        return canvas
    
    def create_solution_table(self) -> QTableWidget:
        """创建解列表表格"""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["序号", "F1 (Makespan)", "F2 (Labor)", "F3 (Energy)", "综合值", "最优"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.itemSelectionChanged.connect(self.on_solution_selected)
        return table
    
    def append_log(self, msg: str):
        """追加日志"""
        self.log_text.append(msg)
    
    def start_optimization(self):
        """开始优化"""
        self.append_log(f"[{datetime.now().strftime('%H:%M:%S')}] 开始优化...")
        self.append_log(f"问题规模: 工件数={self.problem.n_jobs}, 阶段数={self.problem.n_stages}")
        
        self.cancel_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        
        self.worker = OptimizationWorker(self.problem, self.params)
        self.worker.nsga2_progress.connect(self.on_nsga2_progress)
        self.worker.mosa_progress.connect(self.on_mosa_progress)
        self.worker.log.connect(self.append_log)
        self.worker.nsga2_finished.connect(self.on_nsga2_finished)
        self.worker.mosa_finished.connect(self.on_mosa_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def cancel_optimization(self):
        """取消优化"""
        if self.worker:
            self.worker.cancel()
            self.append_log("⚠️ 正在取消...")
    
    def on_nsga2_progress(self, current: int, total: int, message: str):
        """NSGA-II进度更新"""
        self.nsga2_bar.setValue(current)
        self.nsga2_label.setText(message)
    
    def on_mosa_progress(self, current: int, total: int, message: str):
        """MOSA进度更新"""
        self.mosa_bar.setValue(current)
        self.mosa_label.setText(message)
    
    def on_nsga2_finished(self, solutions: list, conv_data: dict):
        """NSGA-II 完成"""
        self.convergence_data['NSGA-II'] = conv_data
        self.update_convergence_plot()
    
    def on_mosa_finished(self, solutions: list, conv_data: dict):
        """MOSA 完成"""
        self.pareto_solutions = solutions
        self.convergence_data['MOSA'] = conv_data
        
        self.update_convergence_plot()
        self.update_pareto_plot()
        self.update_pareto_2d_plot()
        self.update_solution_table()
        
        # 绘制最优解的甘特图
        if solutions and self.best_solution_idx >= 0:
            self.draw_gantt(solutions[self.best_solution_idx])

    
    def on_error(self, msg: str):
        """错误"""
        QMessageBox.critical(self, "错误", msg)
        self.append_log(f"❌ {msg}")
    
    def on_finished(self):
        """完成"""
        self.cancel_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.nsga2_label.setText("✅ 完成")
        self.mosa_label.setText("✅ 完成")
    
    def update_convergence_plot(self):
        """更新收敛曲线"""
        fig = self.convergence_canvas.figure
        fig.clear()
        
        if 'NSGA-II' in self.convergence_data:
            data = self.convergence_data['NSGA-II']
            
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(data['generation'], data['best_makespan'], 'b-', linewidth=1.5)
            ax1.set_title('NSGA-II: Best Makespan', fontsize=10)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Makespan (min)')
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(data['generation'], data['best_labor_cost'], 'g-', linewidth=1.5)
            ax2.set_title('NSGA-II: Best Labor Cost', fontsize=10)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Cost')
            ax2.grid(True, alpha=0.3)
            
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(data['generation'], data['best_energy'], 'r-', linewidth=1.5)
            ax3.set_title('NSGA-II: Best Energy', fontsize=10)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Energy (kWh)')
            ax3.grid(True, alpha=0.3)
            
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.plot(data['generation'], data['n_pareto'], 'm-', linewidth=1.5)
            ax4.set_title('Pareto Front Size', fontsize=10)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Count')
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.convergence_canvas.draw()
    
    def update_pareto_plot(self):
        """更新 3D Pareto 图"""
        fig = self.pareto_canvas.figure
        fig.clear()
        
        if not self.pareto_solutions:
            return
        
        ax = fig.add_subplot(111, projection='3d')
        
        objectives = np.array([s.objectives for s in self.pareto_solutions])
        
        # 画所有点
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c='blue', s=50, alpha=0.7, edgecolors='white', linewidth=0.5,
                   label='Pareto Solutions')
        
        # 计算综合最优 (简单加权和: F1*w1 + F2*w2 + F3*w3)
        weights = np.array(self.params['weights'])
        scores = np.dot(objectives, weights)  # 直接加权和，不归一化
        best_idx = np.argmin(scores)
        self.best_solution_idx = best_idx
        
        best = objectives[best_idx]
        
        # 标记最优解
        ax.scatter([best[0]], [best[1]], [best[2]],
                   c='red', s=200, marker='*', label='Best Solution')
        
        # 添加注释显示最优解的值
        ax.text(best[0], best[1], best[2],
                f'\n  F1={best[0]:.1f}\n  F2={best[1]:.1f}\n  F3={best[2]:.2f}',
                fontsize=9, color='red')
        
        ax.set_xlabel('F1: Makespan (min)')
        ax.set_ylabel('F2: Labor Cost')
        ax.set_zlabel('F3: Energy (kWh)')
        ax.set_title('Pareto Front (3D)')
        ax.legend(loc='upper left')
        
        fig.tight_layout()
        self.pareto_canvas.draw()
    
    def update_pareto_2d_plot(self):
        """更新 2D Pareto 对比图"""
        fig = self.pareto_2d_canvas.figure
        fig.clear()
        
        if not self.pareto_solutions:
            return
        
        objectives = np.array([s.objectives for s in self.pareto_solutions])
        
        # 计算最优解 (简单加权和: F1*w1 + F2*w2 + F3*w3)
        weights = np.array(self.params['weights'])
        scores = np.dot(objectives, weights)  # 直接加权和，不归一化
        best_idx = np.argmin(scores)
        best = objectives[best_idx]
        
        # F1 vs F2
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(objectives[:, 0], objectives[:, 1], c='blue', s=30, alpha=0.7, label='Pareto')
        ax1.scatter([best[0]], [best[1]], c='red', s=100, marker='*', label='Best')
        ax1.set_xlabel('F1: Makespan (min)')
        ax1.set_ylabel('F2: Labor Cost')
        ax1.set_title('F1 vs F2')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # F1 vs F3
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(objectives[:, 0], objectives[:, 2], c='green', s=30, alpha=0.7, label='Pareto')
        ax2.scatter([best[0]], [best[2]], c='red', s=100, marker='*', label='Best')
        ax2.set_xlabel('F1: Makespan (min)')
        ax2.set_ylabel('F3: Energy (kWh)')
        ax2.set_title('F1 vs F3')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # F2 vs F3
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(objectives[:, 1], objectives[:, 2], c='purple', s=30, alpha=0.7, label='Pareto')
        ax3.scatter([best[1]], [best[2]], c='red', s=100, marker='*', label='Best')
        ax3.set_xlabel('F2: Labor Cost')
        ax3.set_ylabel('F3: Energy (kWh)')
        ax3.set_title('F2 vs F3')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # 最优解信息
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        info_text = f"""Best Solution (Weighted):
        
F1 (Makespan): {best[0]:.2f} min
F2 (Labor Cost): {best[1]:.2f}
F3 (Energy): {best[2]:.2f} kWh

Weights: ({self.params['weights'][0]:.2f}, {self.params['weights'][1]:.2f}, {self.params['weights'][2]:.2f})
Score: {scores[best_idx]:.4f}
"""
        ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        fig.tight_layout()
        self.pareto_2d_canvas.draw()
    
    def update_solution_table(self):
        """更新解列表 - 按综合值排序"""
        if not self.pareto_solutions:
            return
        
        weights = np.array(self.params['weights'])
        objectives = np.array([s.objectives for s in self.pareto_solutions])
        
        # 计算每个解的综合值 (简单加权和: F1*w1 + F2*w2 + F3*w3)
        solution_data = []
        for i, sol in enumerate(self.pareto_solutions):
            obj = np.array(sol.objectives)
            score = np.dot(obj, weights)  # 直接加权和，不归一化
            solution_data.append({
                'idx': i,
                'f1': sol.objectives[0],
                'f2': sol.objectives[1],
                'f3': sol.objectives[2],
                'score': score,
                'solution': sol
            })
        
        # 排序: 综合值 -> 完工时间 -> 能耗
        solution_data.sort(key=lambda x: (x['score'], x['f1'], x['f3']))
        
        # 找最小综合值
        min_score = solution_data[0]['score']

        
        self.solution_table.setRowCount(len(solution_data))
        
        for row, data in enumerate(solution_data):
            # 序号
            self.solution_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            
            # 目标值
            self.solution_table.setItem(row, 1, QTableWidgetItem(f"{data['f1']:.2f}"))
            self.solution_table.setItem(row, 2, QTableWidgetItem(f"{data['f2']:.2f}"))
            self.solution_table.setItem(row, 3, QTableWidgetItem(f"{data['f3']:.2f}"))
            self.solution_table.setItem(row, 4, QTableWidgetItem(f"{data['score']:.4f}"))
            
            # 最优标记
            is_best = abs(data['score'] - min_score) < 1e-6
            best_mark = "⭐" if is_best else ""
            self.solution_table.setItem(row, 5, QTableWidgetItem(best_mark))
            
            # 高亮最优行
            if is_best:
                for col in range(6):
                    item = self.solution_table.item(row, col)
                    if item:
                        item.setBackground(QBrush(QColor(255, 255, 200)))
        
        # 保存排序后的解列表供甘特图使用
        self.sorted_solutions = solution_data
        
        # 更新最优解索引
        self.best_solution_idx = solution_data[0]['idx']
    
    def on_solution_selected(self):
        """选中解时绘制甘特图"""
        rows = self.solution_table.selectionModel().selectedRows()
        if rows and hasattr(self, 'sorted_solutions'):
            row_idx = rows[0].row()
            if row_idx < len(self.sorted_solutions):
                sol = self.sorted_solutions[row_idx]['solution']
                self.draw_gantt(sol)
                self.tabs.setCurrentIndex(2)
    
    def draw_gantt(self, solution: Solution):
        """绘制甘特图"""
        fig = self.gantt_canvas.figure
        fig.clear()
        
        decoder = Decoder(self.problem)
        _, schedule = decoder.decode_with_schedule(solution)
        
        ax = fig.add_subplot(111)
        
        # 颜色映射 - 使用多种颜色
        base_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8B500', '#00CED1', '#FF69B4', '#32CD32', '#FF8C00',
            '#9370DB', '#20B2AA', '#FFB6C1', '#87CEEB', '#DEB887'
        ]
        
        # 填充样式 - 用于区分更多工件
        hatch_patterns = ['', '///', '...', 'xxx', '\\\\\\', '|||', '---', '+++', 'ooo', 'OOO']
        
        # 为每个工件分配颜色和填充样式
        n_jobs = self.problem.n_jobs
        job_styles = []
        for i in range(n_jobs):
            color = base_colors[i % len(base_colors)]
            hatch = hatch_patterns[(i // len(base_colors)) % len(hatch_patterns)]
            job_styles.append({'color': color, 'hatch': hatch})
        
        # 机器标签
        y_pos = 0
        y_ticks = []
        y_labels = []
        
        # 统计使用的工人数量
        workers_used = set()
        
        for stage in range(self.problem.n_stages):
            for machine in range(self.problem.machines_per_stage[stage]):
                y_labels.append(f"S{stage+1}-M{machine+1}")
                y_ticks.append(y_pos)
                
                ops = schedule['machine_utilization'].get((stage, machine), [])
                for op in ops:
                    job = op['job']
                    start = op['start']
                    setup_end = op['setup_end']
                    end = op['end']
                    skill = op['skill']
                    speed = op.get('speed', 0)  # 获取速度等级
                    worker_idx = op.get('worker_idx', 0)  # 获取工人索引
                    
                    # 统计唯一工人：只需要技能等级和工人编号
                    workers_used.add((skill, worker_idx))
                    
                    # 计算工人编号: 等级字母 + 工人序号 (A1, A2, B1, B2, ...)
                    skill_letters = ['A', 'B', 'C', 'D', 'E']
                    skill_letter = skill_letters[skill] if skill < len(skill_letters) else chr(ord('A') + skill)
                    worker_label = f"{skill_letter}{worker_idx + 1}"
                    
                    # 换模时间 (灰色斜线)
                    if setup_end > start:
                        ax.barh(y_pos, setup_end - start, left=start, height=0.6,
                                color='gray', alpha=0.5, hatch='//', edgecolor='black', linewidth=0.3)
                    
                    # 加工时间 - 使用工件特定的颜色和填充样式
                    style = job_styles[job]
                    ax.barh(y_pos, end - setup_end, left=setup_end, height=0.6,
                            color=style['color'], hatch=style['hatch'], 
                            edgecolor='black', linewidth=0.5)
                    
                    # 标注: 工件号(工人编号,速度等级)
                    mid = (setup_end + end) / 2
                    ax.text(mid, y_pos, f"J{job+1}({worker_label},s{speed+1})", 
                            ha='center', va='center', fontsize=6, color='black')
                
                y_pos += 1


        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time (min)')
        
        n_workers = len(workers_used)
        ax.set_title(f'Gantt Chart | Makespan={solution.objectives[0]:.1f} min | Workers Used={n_workers}')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 图例 - 为所有工件生成
        from matplotlib.patches import Patch
        legend_elements = []
        for i in range(n_jobs):
            style = job_styles[i]
            legend_elements.append(Patch(facecolor=style['color'], hatch=style['hatch'],
                                        edgecolor='black', linewidth=0.5, label=f'Job {i+1}'))
        legend_elements.append(Patch(facecolor='gray', hatch='//', edgecolor='black', label='Setup'))
        
        # 根据工件数量调整图例位置和大小
        if n_jobs <= 10:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=1)
        elif n_jobs <= 20:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=6, ncol=2)
        else:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=5, ncol=3)
        
        fig.tight_layout()
        self.gantt_canvas.draw()
    
    def save_figures(self):
        """保存图表"""
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not folder:
            return
        
        try:
            # 保存收敛曲线
            self.convergence_canvas.figure.savefig(
                os.path.join(folder, 'convergence.png'), dpi=300, bbox_inches='tight')
            
            # 保存 Pareto 3D 图
            self.pareto_canvas.figure.savefig(
                os.path.join(folder, 'pareto_3d.png'), dpi=300, bbox_inches='tight')
            
            # 保存 Pareto 2D 对比图
            self.pareto_2d_canvas.figure.savefig(
                os.path.join(folder, 'pareto_2d.png'), dpi=300, bbox_inches='tight')
            
            # 保存甘特图
            self.gantt_canvas.figure.savefig(
                os.path.join(folder, 'gantt.png'), dpi=300, bbox_inches='tight')
            
            # 保存解数据 CSV
            if self.pareto_solutions:
                import csv
                with open(os.path.join(folder, 'pareto_solutions.csv'), 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Rank', 'Makespan', 'LaborCost', 'Energy', 'Score', 'IsBest'])
                    
                    if hasattr(self, 'sorted_solutions'):
                        min_score = self.sorted_solutions[0]['score']
                        for i, data in enumerate(self.sorted_solutions):
                            is_best = "Yes" if abs(data['score'] - min_score) < 1e-6 else "No"
                            writer.writerow([i+1, data['f1'], data['f2'], data['f3'], data['score'], is_best])
            
            QMessageBox.information(self, "成功", f"图表已保存到:\n{folder}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
