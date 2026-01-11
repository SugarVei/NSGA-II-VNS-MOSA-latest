# -*- coding: utf-8 -*-
"""
案例配置对话框
Case Configuration Dialog

提供单个算例的详细配置界面：
- Tab1: 实例数据（加工时间、换模时间、机器能耗、工人参数）
- Tab2: 8个算法的参数编辑
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QScrollArea, QHeaderView, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.problem import SchedulingProblem
from ui.case_data import CaseConfig, get_default_algorithm_params


# 样式
DIALOG_STYLE = """
QDialog {
    background-color: #F1F8E9;
    font-family: "Microsoft YaHei", sans-serif;
}
QTabWidget::pane {
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    background-color: white;
}
QTabBar::tab {
    background-color: #C8E6C9;
    color: #1B5E20;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #43A047;
    color: white;
}
QTableWidget {
    background-color: white;
    border: 1px solid #A5D6A7;
    gridline-color: #E8F5E9;
    font-size: 12px;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #66BB6A;
    color: white;
    padding: 6px;
    border: none;
    font-weight: bold;
    font-size: 11px;
}
QGroupBox {
    font-size: 14px;
    font-weight: bold;
    color: #1B5E20;
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #43A047;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #388E3C;
}
QSpinBox, QDoubleSpinBox {
    padding: 4px;
    border: 1px solid #A5D6A7;
    border-radius: 4px;
}
QLabel {
    color: #1B5E20;
}
QScrollArea {
    border: none;
}
"""


class CaseConfigDialog(QDialog):
    """
    案例配置对话框
    
    功能：
    - Tab1: 实例数据编辑（加工时间、换模时间、机器能耗、工人参数）
    - Tab2: 8个算法参数编辑
    """
    
    def __init__(self, parent, case: CaseConfig, mode: str = 'auto'):
        """
        初始化对话框
        
        Args:
            parent: 父窗口
            case: 要配置的算例
            mode: 'auto' 自动生成数据, 'manual' 手动输入
        """
        super().__init__(parent)
        self.case = CaseConfig(
            case_no=case.case_no,
            n_jobs=case.n_jobs,
            machines_per_stage=case.machines_per_stage.copy(),
            workers_available=case.workers_available.copy(),
            input_mode=mode,
            problem_data=case.problem_data.copy() if case.problem_data else None,
            algorithm_params=case.algorithm_params.copy() if case.algorithm_params else get_default_algorithm_params(),
            is_configured=case.is_configured
        )
        self.mode = mode
        self.problem = None  # SchedulingProblem 实例
        
        # 设置窗口支持最大化
        self.setWindowFlags(
            Qt.Window | 
            Qt.WindowMinimizeButtonHint | 
            Qt.WindowMaximizeButtonHint | 
            Qt.WindowCloseButtonHint
        )
        
        # 数据表格引用
        self.processing_tables = []  # 加工时间表格
        self.setup_tables = []       # 换模时间表格
        self.energy_tables = []      # 能耗表格
        self.worker_table = None     # 工人参数表格
        self.algorithm_inputs = {}   # 算法参数输入
        
        self.setup_ui()
        self.setStyleSheet(DIALOG_STYLE)
        
        # 初始化数据
        if mode == 'auto':
            self.generate_random_data()
        elif case.problem_data:
            self.load_existing_data()
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle(
            f"算例 {self.case.case_no} 配置 - "
            f"{'自动生成' if self.mode == 'auto' else '手动输入'}"
        )
        self.setMinimumSize(1100, 800)
        self.resize(1200, 900)
        
        layout = QVBoxLayout(self)
        
        # ===== 基本信息 =====
        info_group = QGroupBox("算例信息")
        info_layout = QHBoxLayout(info_group)
        info_layout.addWidget(QLabel(f"序号: {self.case.case_no}"))
        info_layout.addWidget(QLabel(f"问题规模: {self.case.problem_scale_str}"))
        info_layout.addWidget(QLabel(f"机器分布: {self.case.machines_dist_str}"))
        info_layout.addWidget(QLabel(f"工件数: {self.case.n_jobs}"))
        info_layout.addWidget(QLabel(f"工人分布: {self.case.workers_dist_str}"))
        info_layout.addStretch()
        layout.addWidget(info_group)
        
        # ===== Tab 页 =====
        self.tab_widget = QTabWidget()
        
        # Tab1: 实例数据（使用子Tab）
        self.data_tab = self._create_data_tab()
        self.tab_widget.addTab(self.data_tab, "实例数据")
        
        # Tab2: 算法参数
        self.params_tab = self._create_params_tab()
        self.tab_widget.addTab(self.params_tab, "算法参数")
        
        layout.addWidget(self.tab_widget)
        
        # ===== 底部按钮 =====
        btn_layout = QHBoxLayout()
        
        if self.mode == 'auto':
            regenerate_btn = QPushButton("重新生成数据")
            regenerate_btn.clicked.connect(self.generate_random_data)
            btn_layout.addWidget(regenerate_btn)
        
        fill_btn = QPushButton("填充默认值")
        fill_btn.clicked.connect(self.fill_default_values)
        btn_layout.addWidget(fill_btn)
        
        btn_layout.addStretch()
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_and_accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _calculate_table_height(self, row_count: int) -> int:
        """计算表格合适的高度（已适配增大后的行高）"""
        row_height = 56  # 增大后的行高
        header_height = 40
        padding = 10
        return min(row_count * row_height + header_height + padding, 500)
    
    def _create_data_tab(self) -> QWidget:
        """创建实例数据Tab（包含子Tab）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 使用子TabWidget
        self.data_sub_tabs = QTabWidget()
        
        # 子Tab 1: 加工时间
        self.data_sub_tabs.addTab(self._create_processing_time_tab(), "加工时间")
        
        # 子Tab 2: 换模时间
        self.data_sub_tabs.addTab(self._create_setup_time_tab(), "换模时间")
        
        # 子Tab 3: 机器能耗
        self.data_sub_tabs.addTab(self._create_energy_tab(), "机器能耗")
        
        # 子Tab 4: 工人参数
        self.data_sub_tabs.addTab(self._create_worker_tab(), "工人参数")
        
        layout.addWidget(self.data_sub_tabs)
        return widget
    
    def _create_processing_time_tab(self) -> QWidget:
        """创建加工时间输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 说明
        desc = QLabel(
            "输入每个工件在每个阶段、每台机器上的基础加工时间(分钟)。"
            " 速度等级自动调整: 低速=100%, 中速=75%, 高速=50%"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(desc)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        self.processing_tables = []
        
        for stage in range(3):
            n_machines = self.case.machines_per_stage[stage]
            group = QGroupBox(f"阶段 {stage + 1} (共 {n_machines} 台机器)")
            group_layout = QVBoxLayout(group)
            
            table = QTableWidget()
            table.setRowCount(self.case.n_jobs)
            table.setColumnCount(n_machines)
            
            table.setHorizontalHeaderLabels([f"机器{m+1}" for m in range(n_machines)])
            table.setVerticalHeaderLabels([f"工件{j+1}" for j in range(self.case.n_jobs)])
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setDefaultSectionSize(56)
            
            table_height = self._calculate_table_height(self.case.n_jobs)
            table.setMinimumHeight(table_height)
            
            # 初始化默认值
            for i in range(self.case.n_jobs):
                for j in range(n_machines):
                    item = QTableWidgetItem("30")
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, j, item)
            
            group_layout.addWidget(table)
            scroll_layout.addWidget(group)
            self.processing_tables.append((stage, n_machines, table))
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_setup_time_tab(self) -> QWidget:
        """创建换模时间输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        desc = QLabel(
            "输入机器在处理不同工件时的换模/设置时间(分钟)。"
            " 对角线为0(同工件无需换模)。"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(desc)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        self.setup_tables = []
        
        for stage in range(3):
            for machine in range(self.case.machines_per_stage[stage]):
                group = QGroupBox(f"阶段{stage+1} - 机器{machine+1}")
                group_layout = QVBoxLayout(group)
                
                table = QTableWidget()
                table.setRowCount(self.case.n_jobs)
                table.setColumnCount(self.case.n_jobs)
                
                table.setHorizontalHeaderLabels([f"→J{j+1}" for j in range(self.case.n_jobs)])
                table.setVerticalHeaderLabels([f"J{i+1}→" for i in range(self.case.n_jobs)])
                
                table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                table.verticalHeader().setDefaultSectionSize(50)
                
                table_height = self._calculate_table_height(self.case.n_jobs)
                table.setMinimumHeight(min(table_height, 200))
                table.setMaximumHeight(250)
                
                # 初始化 (对角线为0)
                for i in range(self.case.n_jobs):
                    for j in range(self.case.n_jobs):
                        value = "0" if i == j else "5"
                        item = QTableWidgetItem(value)
                        item.setTextAlignment(Qt.AlignCenter)
                        if i == j:
                            item.setBackground(Qt.lightGray)
                        table.setItem(i, j, item)
                
                group_layout.addWidget(table)
                scroll_layout.addWidget(group)
                self.setup_tables.append((stage, machine, table))
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_energy_tab(self) -> QWidget:
        """创建机器能耗输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        desc = QLabel(
            "输入每台机器的功率参数(kW)。"
            " 加工功率按速度等级区分，换模功率和空闲功率为固定值。"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(desc)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        self.energy_tables = []
        
        for stage in range(3):
            n_machines = self.case.machines_per_stage[stage]
            group = QGroupBox(f"阶段 {stage + 1} - 机器能耗参数(kW)")
            group_layout = QVBoxLayout(group)
            
            table = QTableWidget()
            table.setRowCount(n_machines)
            table.setColumnCount(5)
            
            # 5列: 低速功率, 中速功率, 高速功率, 换模功率, 空闲功率
            table.setHorizontalHeaderLabels([
                "加工(低速)", "加工(中速)", "加工(高速)", "换模功率", "空闲功率"
            ])
            table.setVerticalHeaderLabels([f"机器{m+1}" for m in range(n_machines)])
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setDefaultSectionSize(56)
            
            table_height = self._calculate_table_height(n_machines)
            table.setMinimumHeight(table_height)
            table.setMaximumHeight(200)
            
            # 初始化默认值
            for m in range(n_machines):
                base_power = 5.0 + m * 0.5
                for s in range(3):
                    power = base_power * (1.0 + 0.5 * s)
                    item = QTableWidgetItem(f"{power:.1f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(m, s, item)
                
                # 换模功率
                item = QTableWidgetItem(f"{base_power * 0.7:.1f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(m, 3, item)
                
                # 空闲功率
                item = QTableWidgetItem(f"{base_power * 0.15:.1f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(m, 4, item)
            
            group_layout.addWidget(table)
            scroll_layout.addWidget(group)
            self.energy_tables.append((stage, n_machines, table))
        
        # 固定能耗参数
        fixed_group = QGroupBox("固定能耗参数")
        fixed_layout = QGridLayout(fixed_group)
        
        fixed_layout.addWidget(QLabel("运输功率 (kW):"), 0, 0)
        self.transport_power_spin = QDoubleSpinBox()
        self.transport_power_spin.setRange(0, 10)
        self.transport_power_spin.setValue(0.5)
        self.transport_power_spin.setDecimals(2)
        fixed_layout.addWidget(self.transport_power_spin, 0, 1)
        
        fixed_layout.addWidget(QLabel("辅助功率 (kW):"), 0, 2)
        self.aux_power_spin = QDoubleSpinBox()
        self.aux_power_spin.setRange(0, 10)
        self.aux_power_spin.setValue(1.0)
        self.aux_power_spin.setDecimals(2)
        fixed_layout.addWidget(self.aux_power_spin, 0, 3)
        
        scroll_layout.addWidget(fixed_group)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_worker_tab(self) -> QWidget:
        """创建工人参数输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        desc = QLabel(
            "输入不同技能等级工人的工资(元/班次)和可用人数。"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(desc)
        
        group = QGroupBox("工人技能等级设置")
        group_layout = QVBoxLayout(group)
        
        self.worker_table = QTableWidget()
        self.worker_table.setRowCount(3)
        self.worker_table.setColumnCount(3)
        
        self.worker_table.setHorizontalHeaderLabels(["工资(元/班次)", "可用人数", "可操作最高速度"])
        self.worker_table.setVerticalHeaderLabels(["初级(L1)", "中级(L2)", "高级(L3)"])
        
        self.worker_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.worker_table.verticalHeader().setDefaultSectionSize(70)
        
        table_height = self._calculate_table_height(3)
        self.worker_table.setMinimumHeight(table_height)
        self.worker_table.setMaximumHeight(180)
        
        # 默认值
        default_wages = [150, 225, 300]
        speed_names = ["低速", "中速", "高速"]
        
        for s in range(3):
            # 工资
            item = QTableWidgetItem(str(default_wages[s]))
            item.setTextAlignment(Qt.AlignCenter)
            self.worker_table.setItem(s, 0, item)
            
            # 可用人数
            item = QTableWidgetItem(str(self.case.workers_available[s]))
            item.setTextAlignment(Qt.AlignCenter)
            self.worker_table.setItem(s, 1, item)
            
            # 可操作最高速度
            item = QTableWidgetItem(speed_names[s])
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setBackground(Qt.lightGray)
            self.worker_table.setItem(s, 2, item)
        
        group_layout.addWidget(self.worker_table)
        layout.addWidget(group)
        layout.addStretch()
        
        return widget
    
    def _create_params_tab(self) -> QWidget:
        """创建算法参数Tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 每个算法一个 GroupBox
        algorithms = [
            'NSGA-II', 'MOEA/D', 'SPEA2', 'MOPSO',
            'MOSA', 'NSGA2-VNS', 'NSGA2-MOSA', 'NSGA2-VNS-MOSA'
        ]
        
        for alg_name in algorithms:
            params = self.case.algorithm_params.get(alg_name, {})
            group = QGroupBox(alg_name)
            grid = QGridLayout(group)
            
            self.algorithm_inputs[alg_name] = {}
            row = 0
            col = 0
            
            for param_name, value in params.items():
                label = QLabel(param_name)
                grid.addWidget(label, row, col * 2)
                
                if isinstance(value, float):
                    spin = QDoubleSpinBox()
                    spin.setRange(0.0, 10000.0)
                    spin.setDecimals(3)
                    spin.setValue(value)
                else:
                    spin = QSpinBox()
                    spin.setRange(1, 10000)
                    spin.setValue(int(value))
                
                grid.addWidget(spin, row, col * 2 + 1)
                self.algorithm_inputs[alg_name][param_name] = spin
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
            
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return widget
    
    def generate_random_data(self):
        """自动生成随机数据"""
        seed = self.case.case_no * 1000 + 42
        
        self.problem = SchedulingProblem.generate_random(
            n_jobs=self.case.n_jobs,
            n_stages=3,
            machines_per_stage=self.case.machines_per_stage,
            n_speed_levels=3,
            n_skill_levels=3,
            workers_available=self.case.workers_available,
            seed=seed
        )
        
        self._fill_data_from_problem()
    
    def fill_default_values(self):
        """填充默认值"""
        # 加工时间
        for stage, n_machines, table in self.processing_tables:
            for i in range(table.rowCount()):
                for j in range(n_machines):
                    value = np.random.randint(15, 60)
                    item = table.item(i, j)
                    if item:
                        item.setText(str(value))
        
        # 换模时间
        for stage, machine, table in self.setup_tables:
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    if i == j:
                        table.item(i, j).setText("0")
                    else:
                        value = np.random.randint(3, 10)
                        table.item(i, j).setText(str(value))
        
        QMessageBox.information(self, "提示", "已填充默认值！")
    
    def load_existing_data(self):
        """加载现有数据"""
        if not self.case.problem_data:
            return
        
        pd = self.case.problem_data
        workers = pd.get('workers_available_arr')
        if workers is None:
            workers = np.array(self.case.workers_available)
        
        self.problem = SchedulingProblem(
            n_jobs=self.case.n_jobs,
            n_stages=3,
            machines_per_stage=self.case.machines_per_stage,
            n_speed_levels=3,
            n_skill_levels=3,
            processing_time=pd.get('processing_time'),
            setup_time=pd.get('setup_time'),
            transport_time=pd.get('transport_time'),
            processing_power=pd.get('processing_power'),
            setup_power=pd.get('setup_power'),
            idle_power=pd.get('idle_power'),
            transport_power=pd.get('transport_power', 0.5),
            aux_power=pd.get('aux_power', 1.0),
            skill_wages=pd.get('skill_wages'),
            workers_available=workers
        )
        
        self._fill_data_from_problem()
    
    def _fill_data_from_problem(self):
        """从 SchedulingProblem 填充表格数据"""
        if not self.problem:
            return
        
        # 填充加工时间表格
        for stage, n_machines, table in self.processing_tables:
            for job in range(self.case.n_jobs):
                for m in range(n_machines):
                    time_val = self.problem.processing_time[job, stage, m, 0]  # 基础时间(低速)
                    item = table.item(job, m)
                    if item:
                        item.setText(f"{time_val:.1f}")
                    else:
                        item = QTableWidgetItem(f"{time_val:.1f}")
                        item.setTextAlignment(Qt.AlignCenter)
                        table.setItem(job, m, item)
        
        # 填充换模时间表格
        for stage, machine, table in self.setup_tables:
            for i in range(self.case.n_jobs):
                for j in range(self.case.n_jobs):
                    time_val = self.problem.setup_time[stage, machine, i, j]
                    item = table.item(i, j)
                    if item:
                        item.setText(f"{time_val:.1f}")
        
        # 填充能耗表格
        for stage, n_machines, table in self.energy_tables:
            for m in range(n_machines):
                for s in range(3):
                    power = self.problem.processing_power[stage, m, s]
                    item = table.item(m, s)
                    if item:
                        item.setText(f"{power:.2f}")
                
                # 换模功率
                setup_p = self.problem.setup_power[stage, m]
                item = table.item(m, 3)
                if item:
                    item.setText(f"{setup_p:.2f}")
                
                # 空闲功率
                idle_p = self.problem.idle_power[stage, m]
                item = table.item(m, 4)
                if item:
                    item.setText(f"{idle_p:.2f}")
        
        # 填充固定能耗
        self.transport_power_spin.setValue(self.problem.transport_power)
        self.aux_power_spin.setValue(self.problem.aux_power)
        
        # 填充工人参数
        for i in range(3):
            self.worker_table.item(i, 0).setText(str(int(self.problem.skill_wages[i])))
            self.worker_table.item(i, 1).setText(str(int(self.problem.workers_available[i])))
    
    def _collect_data_to_problem_data(self) -> dict:
        """从表格收集数据到 problem_data 字典"""
        max_machines = max(self.case.machines_per_stage)
        
        # 收集加工时间
        processing_time = np.zeros((self.case.n_jobs, 3, max_machines, 3))
        for stage, n_machines, table in self.processing_tables:
            for job in range(self.case.n_jobs):
                for m in range(n_machines):
                    item = table.item(job, m)
                    if item:
                        try:
                            base_time = float(item.text())
                        except:
                            base_time = 30.0
                        for s in range(3):
                            speed_factor = 1.0 - 0.25 * s
                            processing_time[job, stage, m, s] = base_time * speed_factor
        
        # 收集换模时间
        setup_time = np.zeros((3, max_machines, self.case.n_jobs, self.case.n_jobs))
        for stage, machine, table in self.setup_tables:
            for i in range(self.case.n_jobs):
                for j in range(self.case.n_jobs):
                    item = table.item(i, j)
                    if item:
                        try:
                            setup_time[stage, machine, i, j] = float(item.text())
                        except:
                            setup_time[stage, machine, i, j] = 0.0
        
        # 收集能耗
        processing_power = np.zeros((3, max_machines, 3))
        setup_power = np.zeros((3, max_machines))
        idle_power = np.zeros((3, max_machines))
        
        for stage, n_machines, table in self.energy_tables:
            for m in range(n_machines):
                for s in range(3):
                    item = table.item(m, s)
                    if item:
                        try:
                            processing_power[stage, m, s] = float(item.text())
                        except:
                            processing_power[stage, m, s] = 5.0
                
                item = table.item(m, 3)
                setup_power[stage, m] = float(item.text()) if item else 3.0
                
                item = table.item(m, 4)
                idle_power[stage, m] = float(item.text()) if item else 0.5
        
        # 收集工人参数
        workers = []
        wages = []
        for i in range(3):
            count_item = self.worker_table.item(i, 1)
            wage_item = self.worker_table.item(i, 0)
            workers.append(int(float(count_item.text())) if count_item else self.case.workers_available[i])
            wages.append(float(wage_item.text()) if wage_item else 150 + i * 75)
        
        return {
            'processing_time': processing_time,
            'setup_time': setup_time,
            'processing_power': processing_power,
            'setup_power': setup_power,
            'idle_power': idle_power,
            'transport_power': self.transport_power_spin.value(),
            'aux_power': self.aux_power_spin.value(),
            'skill_wages': np.array(wages),
            'workers_available_arr': np.array(workers),
        }
    
    def _collect_algorithm_params(self) -> dict:
        """收集算法参数"""
        params = {}
        for alg_name, inputs in self.algorithm_inputs.items():
            params[alg_name] = {}
            for param_name, spin in inputs.items():
                params[alg_name][param_name] = spin.value()
        return params
    
    def save_and_accept(self):
        """保存并接受"""
        self.case.problem_data = self._collect_data_to_problem_data()
        self.case.algorithm_params = self._collect_algorithm_params()
        self.case.input_mode = self.mode
        self.case.is_configured = True
        
        # 更新 workers_available 列表
        self.case.workers_available = [
            int(float(self.worker_table.item(i, 1).text())) for i in range(3)
        ]
        
        self.accept()
    
    def get_updated_case(self) -> CaseConfig:
        """获取更新后的算例配置"""
        return self.case
