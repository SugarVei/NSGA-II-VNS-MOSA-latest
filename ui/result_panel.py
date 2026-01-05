"""
结果展示面板模块
Result Panel Module

展示优化结果、图表和导出功能。
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QTextEdit,
    QFrame, QSplitter, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from typing import List, Optional, Dict
import numpy as np
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.solution import Solution
from visualization.convergence import plot_convergence, plot_comparison
from visualization.pareto_3d import plot_pareto_3d, plot_pareto_2d_projections
from visualization.export import export_pareto_to_csv, generate_report


class ResultPanel(QWidget):
    """
    结果展示面板
    
    包含图表展示、数值结果和导出功能。
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.pareto_solutions: List[Solution] = []
        self.convergence_data: Dict = {}
        self.current_figures: Dict[str, Figure] = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        # Tab 1: Pareto前沿图
        self.pareto_tab = self._create_pareto_tab()
        self.tab_widget.addTab(self.pareto_tab, "Pareto前沿")
        
        # Tab 2: 收敛曲线
        self.convergence_tab = self._create_convergence_tab()
        self.tab_widget.addTab(self.convergence_tab, "收敛曲线")
        
        # Tab 3: 数值结果
        self.results_tab = self._create_results_tab()
        self.tab_widget.addTab(self.results_tab, "数值结果")
        
        # Tab 4: 日志
        self.log_tab = self._create_log_tab()
        self.tab_widget.addTab(self.log_tab, "运行日志")
        
        layout.addWidget(self.tab_widget)
        
        # 底部导出按钮
        export_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("📊 导出CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        
        self.export_plots_btn = QPushButton("📈 保存图表")
        self.export_plots_btn.clicked.connect(self.export_plots)
        self.export_plots_btn.setEnabled(False)
        
        self.export_report_btn = QPushButton("📄 生成报告")
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_report_btn.setEnabled(False)
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_plots_btn)
        export_layout.addWidget(self.export_report_btn)
        
        layout.addLayout(export_layout)
    
    def _create_pareto_tab(self) -> QWidget:
        """创建Pareto前沿选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 3D图画布
        self.pareto_figure = Figure(figsize=(8, 6))
        self.pareto_canvas = FigureCanvas(self.pareto_figure)
        self.pareto_toolbar = NavigationToolbar(self.pareto_canvas, widget)
        
        layout.addWidget(self.pareto_toolbar)
        layout.addWidget(self.pareto_canvas)
        
        # 视图切换按钮
        view_layout = QHBoxLayout()
        self.view_3d_btn = QPushButton("3D视图")
        self.view_3d_btn.clicked.connect(lambda: self.update_pareto_view('3d'))
        self.view_2d_btn = QPushButton("2D投影")
        self.view_2d_btn.clicked.connect(lambda: self.update_pareto_view('2d'))
        
        view_layout.addWidget(self.view_3d_btn)
        view_layout.addWidget(self.view_2d_btn)
        view_layout.addStretch()
        
        layout.addLayout(view_layout)
        
        return widget
    
    def _create_convergence_tab(self) -> QWidget:
        """创建收敛曲线选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.convergence_figure = Figure(figsize=(10, 8))
        self.convergence_canvas = FigureCanvas(self.convergence_figure)
        self.convergence_toolbar = NavigationToolbar(self.convergence_canvas, widget)
        
        layout.addWidget(self.convergence_toolbar)
        layout.addWidget(self.convergence_canvas)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """创建数值结果选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 统计摘要
        summary_group = QGroupBox("优化结果摘要")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_label = QLabel("等待优化...")
        self.summary_label.setWordWrap(True)
        self.summary_label.setFont(QFont("Consolas", 10))
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        # Pareto解表格
        table_group = QGroupBox("Pareto解集")
        table_layout = QVBoxLayout(table_group)
        
        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(6)
        self.solutions_table.setHorizontalHeaderLabels([
            "编号", "Makespan(分钟)", "人工成本(元)", "能耗(kWh)", "排名", "拥挤度"
        ])
        self.solutions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.solutions_table.setAlternatingRowColors(True)
        
        table_layout.addWidget(self.solutions_table)
        layout.addWidget(table_group)
        
        return widget
    
    def _create_log_tab(self) -> QWidget:
        """创建日志选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        
        layout.addWidget(self.log_text)
        layout.addWidget(clear_btn)
        
        return widget
    
    def update_pareto_solutions(self, solutions: List[Solution], algorithm_name: str = "MOSA"):
        """
        更新Pareto解集并刷新显示
        
        Args:
            solutions: Pareto解列表
            algorithm_name: 算法名称
        """
        self.pareto_solutions = solutions
        
        # 更新3D图
        self.pareto_figure.clear()
        if solutions:
            fig = plot_pareto_3d(solutions, title=f"{algorithm_name} Pareto前沿")
            self._copy_figure(fig, self.pareto_figure)
            plt.close(fig)
        
        self.pareto_canvas.draw()
        self.current_figures['pareto'] = self.pareto_figure
        
        # 更新表格
        self._update_solutions_table(solutions)
        
        # 更新摘要
        self._update_summary(solutions)
        
        # 启用导出按钮
        self.export_csv_btn.setEnabled(bool(solutions))
        self.export_plots_btn.setEnabled(bool(solutions))
        self.export_report_btn.setEnabled(bool(solutions))
    
    def update_pareto_view(self, view_type: str):
        """切换Pareto图视图类型"""
        self.pareto_figure.clear()
        
        if not self.pareto_solutions:
            return
        
        if view_type == '3d':
            fig = plot_pareto_3d(self.pareto_solutions)
        else:
            fig = plot_pareto_2d_projections(self.pareto_solutions)
        
        self._copy_figure(fig, self.pareto_figure)
        plt.close(fig)
        self.pareto_canvas.draw()
    
    def update_convergence(self, data_dict: Dict[str, Dict]):
        """
        更新收敛曲线
        
        Args:
            data_dict: {算法名: 收敛数据} 的字典
        """
        self.convergence_data = data_dict
        
        self.convergence_figure.clear()
        
        if data_dict:
            fig = plot_comparison(data_dict)
            self._copy_figure(fig, self.convergence_figure)
            plt.close(fig)
        
        self.convergence_canvas.draw()
        self.current_figures['convergence'] = self.convergence_figure
    
    def _copy_figure(self, source: Figure, target: Figure):
        """复制图形内容"""
        target.clear()
        
        for ax in source.axes:
            new_ax = target.add_subplot(ax.get_subplotspec(), projection=ax.name if ax.name != 'rectilinear' else None)
            
            # 复制基本属性
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            
            # 复制线条
            for line in ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(),
                           color=line.get_color(),
                           linewidth=line.get_linewidth(),
                           linestyle=line.get_linestyle(),
                           label=line.get_label())
            
            # 复制散点 (简化处理)
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        new_ax.scatter(offsets[:, 0], offsets[:, 1] if offsets.shape[1] > 1 else None,
                                      alpha=0.7)
            
            if ax.get_legend():
                new_ax.legend()
            
            new_ax.grid(True, alpha=0.3)
        
        target.tight_layout()
    
    def _update_solutions_table(self, solutions: List[Solution]):
        """更新解集表格"""
        self.solutions_table.setRowCount(len(solutions))
        
        for i, sol in enumerate(solutions):
            self.solutions_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.solutions_table.setItem(i, 1, QTableWidgetItem(f"{sol.objectives[0]:.2f}"))
            self.solutions_table.setItem(i, 2, QTableWidgetItem(f"{sol.objectives[1]:.2f}"))
            self.solutions_table.setItem(i, 3, QTableWidgetItem(f"{sol.objectives[2]:.2f}"))
            self.solutions_table.setItem(i, 4, QTableWidgetItem(str(sol.rank)))
            self.solutions_table.setItem(i, 5, QTableWidgetItem(f"{sol.crowding_distance:.4f}"))
    
    def _update_summary(self, solutions: List[Solution]):
        """更新结果摘要"""
        if not solutions:
            self.summary_label.setText("无有效解")
            return
        
        objectives = np.array([s.objectives for s in solutions])
        
        summary = f"""
优化完成! 共找到 {len(solutions)} 个Pareto最优解

目标函数统计:
{'='*40}
Makespan (F1):
  最小值: {objectives[:, 0].min():.2f} 分钟
  最大值: {objectives[:, 0].max():.2f} 分钟
  平均值: {objectives[:, 0].mean():.2f} 分钟

人工成本 (F2):
  最小值: {objectives[:, 1].min():.2f} 元
  最大值: {objectives[:, 1].max():.2f} 元
  平均值: {objectives[:, 1].mean():.2f} 元

能耗 (F3):
  最小值: {objectives[:, 2].min():.2f} kWh
  最大值: {objectives[:, 2].max():.2f} kWh
  平均值: {objectives[:, 2].mean():.2f} kWh
"""
        self.summary_label.setText(summary)
    
    def append_log(self, message: str):
        """追加日志消息"""
        self.log_text.append(message)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def export_csv(self):
        """导出CSV文件"""
        if not self.pareto_solutions:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "pareto_solutions.csv", "CSV文件 (*.csv)"
        )
        
        if filepath:
            try:
                export_pareto_to_csv(self.pareto_solutions, filepath, include_decisions=True)
                QMessageBox.information(self, "成功", f"已保存到: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def export_plots(self):
        """导出图表"""
        directory = QFileDialog.getExistingDirectory(self, "选择保存目录")
        
        if directory:
            try:
                # 保存Pareto图
                pareto_path = os.path.join(directory, "pareto_front.png")
                self.pareto_figure.savefig(pareto_path, dpi=150, bbox_inches='tight')
                
                # 保存收敛图
                convergence_path = os.path.join(directory, "convergence.png")
                self.convergence_figure.savefig(convergence_path, dpi=150, bbox_inches='tight')
                
                QMessageBox.information(self, "成功", f"图表已保存到: {directory}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def export_report(self):
        """导出完整报告"""
        directory = QFileDialog.getExistingDirectory(self, "选择报告保存目录")
        
        if directory:
            try:
                files = generate_report(
                    self.pareto_solutions,
                    self.convergence_data.get('MOSA', {}),
                    'MOSA',
                    directory
                )
                
                # 保存图表
                pareto_path = os.path.join(directory, "pareto_front.png")
                self.pareto_figure.savefig(pareto_path, dpi=150, bbox_inches='tight')
                
                convergence_path = os.path.join(directory, "convergence.png")
                self.convergence_figure.savefig(convergence_path, dpi=150, bbox_inches='tight')
                
                QMessageBox.information(self, "成功", 
                    f"报告已生成!\n\n已保存文件:\n- {files.get('pareto_csv', '')}\n- {files.get('summary_txt', '')}\n- pareto_front.png\n- convergence.png")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def clear(self):
        """清空所有结果"""
        self.pareto_solutions = []
        self.convergence_data = {}
        
        self.pareto_figure.clear()
        self.pareto_canvas.draw()
        
        self.convergence_figure.clear()
        self.convergence_canvas.draw()
        
        self.solutions_table.setRowCount(0)
        self.summary_label.setText("等待优化...")
        
        self.export_csv_btn.setEnabled(False)
        self.export_plots_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
