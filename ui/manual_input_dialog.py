"""
手动数据输入对话框模块
Manual Data Input Dialog Module

提供详细的数据输入界面，让用户手动输入:
- 每个阶段每台机器处理工件的时间
- 不同机器处理不同工件的设置时间
- 不同机器的速度
- 不同机器的能耗成本
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QScrollArea, QFrame, QWidget, QHeaderView, QMessageBox,
    QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np


class ManualDataInputDialog(QDialog):
    """
    手动数据输入对话框
    
    让用户输入详细的调度问题数据。
    """
    
    def __init__(self, n_jobs: int, n_stages: int, machines_per_stage: int,
                 n_speed_levels: int, n_skill_levels: int, parent=None):
        super().__init__(parent)
        
        self.n_jobs = n_jobs
        self.n_stages = n_stages
        self.machines_per_stage = machines_per_stage
        self.n_speed_levels = n_speed_levels
        self.n_skill_levels = n_skill_levels
        
        # 数据存储
        self.processing_time_data = None
        self.setup_time_data = None
        self.energy_rate_data = None
        self.speed_factor_data = None
        self.skill_wages_data = None
        
        self.setup_ui()
        
        # 设置对话框大小 - 更大的窗口
        self.resize(1100, 800)
        self.setMinimumSize(900, 600)
        self.setWindowTitle("手动输入数据 - 调度问题参数")
    
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 说明标签 - 更紧凑
        info_label = QLabel(
            f"📝 问题规模: {self.n_jobs}个工件, {self.n_stages}个阶段, "
            f"每阶段{self.machines_per_stage}台机器, "
            f"{self.n_speed_levels}个速度等级"
        )
        info_label.setFont(QFont("Microsoft YaHei", 9))
        info_label.setStyleSheet("color: #1976D2; padding: 6px; background: #E3F2FD; border-radius: 4px;")
        layout.addWidget(info_label)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        # Tab 1: 加工时间
        self.processing_tab = self._create_processing_time_tab()
        self.tab_widget.addTab(self.processing_tab, "加工时间")
        
        # Tab 2: 设置时间
        self.setup_tab = self._create_setup_time_tab()
        self.tab_widget.addTab(self.setup_tab, "设置时间")
        
        # Tab 3: 机器能耗
        self.energy_tab = self._create_energy_tab()
        self.tab_widget.addTab(self.energy_tab, "机器能耗")
        
        # Tab 4: 工人工资
        self.worker_tab = self._create_worker_tab()
        self.tab_widget.addTab(self.worker_tab, "工人工资")
        
        layout.addWidget(self.tab_widget, 1)  # 占据主要空间
        
        # 底部按钮 - 紧凑布局
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.fill_default_btn = QPushButton("填充默认值")
        self.fill_default_btn.clicked.connect(self.fill_default_values)
        
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_all)
        
        btn_layout.addWidget(self.fill_default_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.confirm_btn = QPushButton("确认输入")
        self.confirm_btn.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 8px 20px;")
        self.confirm_btn.clicked.connect(self.validate_and_accept)
        
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.confirm_btn)
        
        layout.addLayout(btn_layout)
    
    def _calculate_table_height(self, row_count: int) -> int:
        """计算表格合适的高度，使其完整显示"""
        row_height = 30  # 每行高度
        header_height = 30  # 表头高度
        padding = 10  # 边距
        return row_count * row_height + header_height + padding
    
    def _create_processing_time_tab(self) -> QWidget:
        """创建加工时间输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 说明 - 更紧凑
        desc = QLabel(
            "输入每个工件在每个阶段、每台机器上的基础加工时间(分钟)。"
            " 速度等级自动调整: 低速=100%, 中速=75%, 高速=50%"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(desc)
        
        # 为每个阶段创建一个表格
        self.processing_tables = []
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        for stage in range(self.n_stages):
            group = QGroupBox(f"阶段 {stage + 1}")
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(5, 10, 5, 5)
            
            table = QTableWidget()
            table.setRowCount(self.n_jobs)
            table.setColumnCount(self.machines_per_stage)
            
            # 设置表头
            table.setHorizontalHeaderLabels([f"机器{m+1}" for m in range(self.machines_per_stage)])
            table.setVerticalHeaderLabels([f"工件{j+1}" for j in range(self.n_jobs)])
            
            # 自适应列宽
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            table.verticalHeader().setDefaultSectionSize(28)
            
            # 设置固定高度，使表格完整显示不需要滚动
            table_height = self._calculate_table_height(self.n_jobs)
            table.setMinimumHeight(table_height)
            table.setMaximumHeight(table_height)
            
            # 初始化默认值
            for i in range(self.n_jobs):
                for j in range(self.machines_per_stage):
                    item = QTableWidgetItem("30")
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(i, j, item)
            
            group_layout.addWidget(table)
            scroll_layout.addWidget(group)
            self.processing_tables.append(table)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_setup_time_tab(self) -> QWidget:
        """创建设置时间输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        desc = QLabel(
            "输入机器在处理不同工件时的设置/切换时间(分钟)。"
            " 对角线为0(同工件无需设置)。"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(desc)
        
        # 为每个阶段的每台机器创建设置时间矩阵
        self.setup_tables = []
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        for stage in range(self.n_stages):
            for machine in range(self.machines_per_stage):
                group = QGroupBox(f"阶段{stage+1} - 机器{machine+1}")
                group.setStyleSheet("QGroupBox { font-weight: bold; }")
                group_layout = QVBoxLayout(group)
                group_layout.setContentsMargins(5, 10, 5, 5)
                
                table = QTableWidget()
                table.setRowCount(self.n_jobs)
                table.setColumnCount(self.n_jobs)
                
                # 设置表头
                table.setHorizontalHeaderLabels([f"→J{j+1}" for j in range(self.n_jobs)])
                table.setVerticalHeaderLabels([f"J{i+1}→" for i in range(self.n_jobs)])
                
                table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
                table.verticalHeader().setDefaultSectionSize(28)
                
                # 设置固定高度
                table_height = self._calculate_table_height(self.n_jobs)
                table.setMinimumHeight(table_height)
                table.setMaximumHeight(table_height)
                
                # 初始化 (对角线为0)
                for i in range(self.n_jobs):
                    for j in range(self.n_jobs):
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
            "输入每台机器在不同速度等级下的能耗率(kW)。"
            " 能耗 = 功率 × 时间"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(desc)
        
        self.energy_tables = []
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        for stage in range(self.n_stages):
            group = QGroupBox(f"阶段 {stage + 1} - 机器能耗率(kW)")
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(5, 10, 5, 5)
            
            table = QTableWidget()
            table.setRowCount(self.machines_per_stage)
            table.setColumnCount(self.n_speed_levels)
            
            # 设置表头
            speed_names = ["低速", "中速", "高速", "超高速", "极速"][:self.n_speed_levels]
            table.setHorizontalHeaderLabels(speed_names)
            table.setVerticalHeaderLabels([f"机器{m+1}" for m in range(self.machines_per_stage)])
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            table.verticalHeader().setDefaultSectionSize(28)
            
            # 设置固定高度
            table_height = self._calculate_table_height(self.machines_per_stage)
            table.setMinimumHeight(table_height)
            table.setMaximumHeight(table_height)
            
            # 初始化默认值
            for m in range(self.machines_per_stage):
                base_power = 5.0 + m * 0.5
                for s in range(self.n_speed_levels):
                    power = base_power * (1.0 + 0.5 * s)
                    item = QTableWidgetItem(f"{power:.1f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(m, s, item)
            
            group_layout.addWidget(table)
            scroll_layout.addWidget(group)
            self.energy_tables.append(table)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_worker_tab(self) -> QWidget:
        """创建工人工资输入选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        desc = QLabel(
            "输入不同技能等级工人的小时工资(元)和可用人数。"
        )
        desc.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(desc)
        
        group = QGroupBox("工人技能等级设置")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(5, 10, 5, 5)
        
        self.worker_table = QTableWidget()
        self.worker_table.setRowCount(self.n_skill_levels)
        self.worker_table.setColumnCount(3)
        
        self.worker_table.setHorizontalHeaderLabels(["小时工资(元)", "可用人数", "可操作最高速度"])
        self.worker_table.setVerticalHeaderLabels([f"技能{s+1}" for s in range(self.n_skill_levels)])
        
        self.worker_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.worker_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.worker_table.verticalHeader().setDefaultSectionSize(28)
        
        # 设置固定高度
        table_height = self._calculate_table_height(self.n_skill_levels)
        self.worker_table.setMinimumHeight(table_height)
        self.worker_table.setMaximumHeight(table_height)
        
        # 初始化默认值
        speed_names = ["低速", "中速", "高速", "超高速", "极速"]
        for s in range(self.n_skill_levels):
            # 工资
            wage = 20 + s * 15
            item = QTableWidgetItem(str(wage))
            item.setTextAlignment(Qt.AlignCenter)
            self.worker_table.setItem(s, 0, item)
            
            # 可用人数
            count = max(5 - s, 1)
            item = QTableWidgetItem(str(count))
            item.setTextAlignment(Qt.AlignCenter)
            self.worker_table.setItem(s, 1, item)
            
            # 可操作最高速度
            speed_name = speed_names[min(s, len(speed_names)-1)]
            item = QTableWidgetItem(speed_name)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setBackground(Qt.lightGray)
            self.worker_table.setItem(s, 2, item)
        
        group_layout.addWidget(self.worker_table)
        layout.addWidget(group)
        layout.addStretch()
        
        return widget
    
    def fill_default_values(self):
        """填充默认值"""
        # 加工时间
        for table in self.processing_tables:
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    value = np.random.randint(15, 60)
                    table.item(i, j).setText(str(value))
        
        # 设置时间
        for stage, machine, table in self.setup_tables:
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    if i == j:
                        table.item(i, j).setText("0")
                    else:
                        value = np.random.randint(3, 10)
                        table.item(i, j).setText(str(value))
        
        QMessageBox.information(self, "提示", "已填充默认值！")
    
    def clear_all(self):
        """清空所有数据"""
        for table in self.processing_tables:
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    table.item(i, j).setText("")
        
        for stage, machine, table in self.setup_tables:
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    if i != j:
                        table.item(i, j).setText("")
    
    def validate_and_accept(self):
        """验证数据并接受"""
        try:
            # 验证加工时间
            processing_time = np.zeros((self.n_jobs, self.n_stages, self.machines_per_stage))
            for stage, table in enumerate(self.processing_tables):
                for job in range(self.n_jobs):
                    for machine in range(self.machines_per_stage):
                        text = table.item(job, machine).text()
                        if not text:
                            raise ValueError(f"阶段{stage+1}工件{job+1}机器{machine+1}的加工时间未填写")
                        value = float(text)
                        if value < 0:
                            raise ValueError(f"加工时间不能为负数")
                        processing_time[job, stage, machine] = value
            
            self.processing_time_data = processing_time
            
            # 验证设置时间
            setup_time = np.zeros((self.n_stages, self.machines_per_stage, self.n_jobs, self.n_jobs))
            for stage, machine, table in self.setup_tables:
                for i in range(self.n_jobs):
                    for j in range(self.n_jobs):
                        text = table.item(i, j).text()
                        if not text:
                            text = "0"
                        value = float(text)
                        if value < 0:
                            raise ValueError(f"设置时间不能为负数")
                        setup_time[stage, machine, i, j] = value
            
            self.setup_time_data = setup_time
            
            # 验证能耗
            energy_rate = np.zeros((self.n_stages, self.machines_per_stage, self.n_speed_levels))
            for stage, table in enumerate(self.energy_tables):
                for machine in range(self.machines_per_stage):
                    for speed in range(self.n_speed_levels):
                        text = table.item(machine, speed).text()
                        if not text:
                            raise ValueError(f"阶段{stage+1}机器{machine+1}速度{speed+1}的能耗未填写")
                        value = float(text)
                        if value < 0:
                            raise ValueError(f"能耗不能为负数")
                        energy_rate[stage, machine, speed] = value
            
            self.energy_rate_data = energy_rate
            
            # 验证工人工资
            skill_wages = np.zeros(self.n_skill_levels)
            workers_available = np.zeros(self.n_skill_levels, dtype=int)
            
            for s in range(self.n_skill_levels):
                wage_text = self.worker_table.item(s, 0).text()
                count_text = self.worker_table.item(s, 1).text()
                
                if not wage_text:
                    raise ValueError(f"技能等级{s+1}的工资未填写")
                if not count_text:
                    raise ValueError(f"技能等级{s+1}的人数未填写")
                
                skill_wages[s] = float(wage_text)
                workers_available[s] = int(float(count_text))
            
            self.skill_wages_data = skill_wages
            self.workers_available_data = workers_available
            
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "数据验证错误", str(e))
    
    def get_data(self) -> dict:
        """
        获取输入的数据
        
        Returns:
            包含所有手动输入数据的字典
        """
        return {
            'processing_time': self.processing_time_data,
            'setup_time': self.setup_time_data,
            'energy_rate': self.energy_rate_data,
            'skill_wages': self.skill_wages_data,
            'workers_available': self.workers_available_data
        }
