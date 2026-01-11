# -*- coding: utf-8 -*-
"""
多算例管理页面
Multi Case Manager Dialog

提供多个算例的表格配置界面，支持：
- 默认显示14个算例
- 新增/删除算例行
- 每行可配置实例数据和算法参数
"""

import sys
import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox, QLabel, QWidget, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.case_data import (
    CaseConfig, get_default_cases, save_cases_config, load_cases_config,
    DEFAULT_CONFIG_PATH, validate_case_config, get_default_algorithm_params
)


# 表格样式
TABLE_STYLE = """
QDialog {
    background-color: #F1F8E9;
    font-family: "Microsoft YaHei", sans-serif;
}
QTableWidget {
    background-color: white;
    border: 2px solid #A5D6A7;
    border-radius: 6px;
    gridline-color: #E8F5E9;
    font-size: 14px;
}
QTableWidget::item {
    padding: 12px 8px;
    min-height: 36px;
}
QHeaderView::section {
    background-color: #43A047;
    color: white;
    padding: 12px 8px;
    border: none;
    font-weight: bold;
}
QPushButton {
    background-color: #43A047;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
    min-width: 80px;
}
QPushButton:hover {
    background-color: #388E3C;
}
QPushButton:disabled {
    background-color: #A5D6A7;
}
QPushButton#autoBtn {
    background-color: #1976D2;
    font-size: 14px;
    padding: 8px 12px;
    min-width: 70px;
}
QPushButton#autoBtn:hover {
    background-color: #1565C0;
}
QPushButton#manualBtn {
    background-color: #F57C00;
    font-size: 14px;
    padding: 8px 12px;
    min-width: 70px;
}
QPushButton#manualBtn:hover {
    background-color: #E65100;
}
QPushButton#deleteBtn {
    background-color: #D32F2F;
    padding: 6px 10px;
    min-width: 55px;
    font-size: 13px;
}
QPushButton#deleteBtn:hover {
    background-color: #B71C1C;
}
QLabel {
    color: #1B5E20;
    font-size: 14px;
}
QGroupBox {
    font-size: 16px;
    font-weight: bold;
    color: #1B5E20;
    border: 2px solid #A5D6A7;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 5px;
}
"""


class MultiCaseManagerDialog(QDialog):
    """
    多算例管理对话框
    
    功能：
    - 表格展示与编辑算例
    - 新增/删除算例行
    - 配置每个算例的实例数据和算法参数
    """
    
    # 表格列定义
    COLUMNS = [
        ('No.', 50),
        ('问题规模', 120),
        ('机器分布', 100),
        ('工件数', 70),
        ('工人分布', 100),
        ('状态', 80),
        ('操作', 250),
    ]
    
    def __init__(self, parent=None, initial_cases=None):
        """
        初始化对话框
        
        Args:
            parent: 父窗口
            initial_cases: 初始算例列表，None 则加载默认或从文件恢复
        """
        super().__init__(parent)
        self.cases = []
        self.case_config_dialog = None  # 延迟导入避免循环依赖
        
        # 设置窗口标志：支持最大化、最小化和关闭按钮
        self.setWindowFlags(
            Qt.Window | 
            Qt.WindowMinimizeButtonHint | 
            Qt.WindowMaximizeButtonHint | 
            Qt.WindowCloseButtonHint
        )
        
        self.setup_ui()
        self.setStyleSheet(TABLE_STYLE)
        
        # 加载初始数据
        if initial_cases:
            self.cases = initial_cases
        else:
            # 尝试从文件加载，否则使用默认
            loaded = load_cases_config(DEFAULT_CONFIG_PATH)
            self.cases = loaded if loaded else get_default_cases()
        
        self.refresh_table()
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle("多问题规模配置")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)  # 设置一个更大的默认尺寸
        
        layout = QVBoxLayout(self)
        
        # ===== 说明文字 =====
        info_label = QLabel(
            '配置多个算例，每个算例可以自动生成数据或手动输入。'
            '点击【自动】或【手动】按钮配置算例详情。'
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # ===== 工具栏 =====
        toolbar = QHBoxLayout()
        
        self.add_btn = QPushButton("+ 新增算例")
        self.add_btn.clicked.connect(self.add_case)
        toolbar.addWidget(self.add_btn)
        
        self.reset_btn = QPushButton("重置为默认")
        self.reset_btn.clicked.connect(self.reset_to_default)
        toolbar.addWidget(self.reset_btn)
        
        # 一键确认配置按钮
        self.auto_all_btn = QPushButton("一键确认配置")
        self.auto_all_btn.setStyleSheet(
            "background-color: #1976D2; font-size: 14px; padding: 8px 16px;"
        )
        self.auto_all_btn.clicked.connect(self.auto_configure_all)
        toolbar.addWidget(self.auto_all_btn)
        
        toolbar.addStretch()
        
        self.count_label = QLabel("共 0 个算例")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        toolbar.addWidget(self.count_label)
        
        layout.addLayout(toolbar)
        
        # ===== 表格 =====
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        
        # 设置列宽
        header = self.table.horizontalHeader()
        for i, (_, width) in enumerate(self.COLUMNS):
            if i == len(self.COLUMNS) - 1:
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                self.table.setColumnWidth(i, width)
        
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        
        layout.addWidget(self.table)
        
        # ===== 底部按钮 =====
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        btn_layout.addWidget(self.save_btn)
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def refresh_table(self):
        """刷新表格数据"""
        self.table.setRowCount(len(self.cases))
        
        # 设置默认行高（增大为原来的2倍，确保文字完整显示）
        self.table.verticalHeader().setDefaultSectionSize(80)
        
        for row, case in enumerate(self.cases):
            # 设置每行的高度（增大为80px）
            self.table.setRowHeight(row, 80)
            
            # No.
            no_item = QTableWidgetItem(str(case.case_no))
            no_item.setTextAlignment(Qt.AlignCenter)
            no_item.setFlags(no_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, no_item)
            
            # 问题规模
            scale_item = QTableWidgetItem(case.problem_scale_str)
            scale_item.setTextAlignment(Qt.AlignCenter)
            scale_item.setFlags(scale_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, scale_item)
            
            # 机器分布
            machines_item = QTableWidgetItem(case.machines_dist_str)
            machines_item.setTextAlignment(Qt.AlignCenter)
            machines_item.setFlags(machines_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, machines_item)
            
            # 工件数
            jobs_item = QTableWidgetItem(str(case.n_jobs))
            jobs_item.setTextAlignment(Qt.AlignCenter)
            jobs_item.setFlags(jobs_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, jobs_item)
            
            # 工人分布
            workers_item = QTableWidgetItem(case.workers_dist_str)
            workers_item.setTextAlignment(Qt.AlignCenter)
            workers_item.setFlags(workers_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 4, workers_item)
            
            # 状态
            status_item = QTableWidgetItem()
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            if case.is_configured:
                status_item.setText("已配置")
                status_item.setBackground(QColor("#C8E6C9"))
            else:
                status_item.setText("未配置")
                status_item.setBackground(QColor("#FFECB3"))
            self.table.setItem(row, 5, status_item)
            
            # 操作按钮
            self._create_action_buttons(row, case)
        
        self.count_label.setText(f"共 {len(self.cases)} 个算例")
    
    def _create_action_buttons(self, row: int, case: CaseConfig):
        """创建操作列的按钮"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        
        # 自动生成按钮
        auto_btn = QPushButton("自动")
        auto_btn.setObjectName("autoBtn")
        auto_btn.setMinimumWidth(60)
        auto_btn.clicked.connect(lambda: self.open_config_dialog(row, 'auto'))
        layout.addWidget(auto_btn)
        
        # 手动输入按钮
        manual_btn = QPushButton("手动")
        manual_btn.setObjectName("manualBtn")
        manual_btn.setMinimumWidth(60)
        manual_btn.clicked.connect(lambda: self.open_config_dialog(row, 'manual'))
        layout.addWidget(manual_btn)
        
        # 删除按钮
        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("deleteBtn")
        delete_btn.setMinimumWidth(55)
        delete_btn.clicked.connect(lambda: self.delete_case(row))
        layout.addWidget(delete_btn)
        
        self.table.setCellWidget(row, 6, widget)
    
    def add_case(self):
        """新增算例行"""
        # 计算新序号
        new_no = max([c.case_no for c in self.cases], default=0) + 1
        
        # 创建新算例，使用上一个算例的参数作为参考，或使用默认值
        if self.cases:
            last = self.cases[-1]
            new_case = CaseConfig(
                case_no=new_no,
                n_jobs=last.n_jobs,
                machines_per_stage=last.machines_per_stage.copy(),
                workers_available=last.workers_available.copy(),
                algorithm_params=get_default_algorithm_params()
            )
        else:
            new_case = CaseConfig(
                case_no=new_no,
                n_jobs=20,
                machines_per_stage=[2, 3, 2],
                workers_available=[4, 3, 1],
                algorithm_params=get_default_algorithm_params()
            )
        
        self.cases.append(new_case)
        self.refresh_table()
        
        # 滚动到新增行
        self.table.scrollToBottom()
    
    def delete_case(self, row: int):
        """删除算例行"""
        if len(self.cases) <= 1:
            QMessageBox.warning(self, "警告", "至少需要保留1个算例")
            return
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除算例 {self.cases[row].case_no} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.cases[row]
            # 重新编号
            for i, case in enumerate(self.cases):
                case.case_no = i + 1
            self.refresh_table()
    
    def reset_to_default(self):
        """重置为默认14个算例"""
        reply = QMessageBox.question(
            self, "确认重置",
            "这将丢失当前所有配置，确定要重置为默认14个算例吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.cases = get_default_cases()
            self.refresh_table()
    
    def auto_configure_all(self):
        """
        一键确认配置：为所有未配置的算例自动生成数据
        """
        from models.problem import SchedulingProblem
        from ui.case_data import get_default_algorithm_params
        
        unconfigured_count = sum(1 for c in self.cases if not c.is_configured)
        
        if unconfigured_count == 0:
            QMessageBox.information(self, "提示", "所有算例已配置完成！")
            return
        
        reply = QMessageBox.question(
            self, "一键确认配置",
            f"将为 {unconfigured_count} 个未配置的算例自动生成数据。\n继续吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 为每个未配置的算例生成数据
        for case in self.cases:
            if case.is_configured:
                continue
            
            # 使用与 CaseConfigDialog.generate_random_data 相同的逻辑
            seed = case.case_no * 1000 + 42
            problem = SchedulingProblem.generate_random(
                n_jobs=case.n_jobs,
                n_stages=3,
                machines_per_stage=case.machines_per_stage,
                n_speed_levels=3,
                n_skill_levels=3,
                workers_available=case.workers_available,
                seed=seed
            )
            
            # 收集问题数据
            import numpy as np
            max_machines = max(case.machines_per_stage)
            
            case.problem_data = {
                'processing_time': problem.processing_time,
                'setup_time': problem.setup_time,
                'processing_power': problem.processing_power,
                'setup_power': problem.setup_power,
                'idle_power': problem.idle_power,
                'transport_power': problem.transport_power,
                'aux_power': problem.aux_power,
                'skill_wages': problem.skill_wages,
                'workers_available_arr': problem.workers_available,
            }
            case.algorithm_params = get_default_algorithm_params()
            case.input_mode = 'auto'
            case.is_configured = True
        
        self.refresh_table()
        QMessageBox.information(
            self, "成功",
            f"已为 {unconfigured_count} 个算例自动生成配置数据！"
        )
    
    def open_config_dialog(self, row: int, mode: str):
        """
        打开算例配置对话框
        
        Args:
            row: 表格行号
            mode: 'auto' 或 'manual'
        """
        # 延迟导入避免循环依赖
        from ui.case_config_dialog import CaseConfigDialog
        
        case = self.cases[row]
        dialog = CaseConfigDialog(self, case, mode)
        
        if dialog.exec_() == QDialog.Accepted:
            # 更新算例配置
            updated_case = dialog.get_updated_case()
            self.cases[row] = updated_case
            self.refresh_table()
    
    def save_config(self):
        """保存配置到文件"""
        if save_cases_config(self.cases, DEFAULT_CONFIG_PATH):
            QMessageBox.information(self, "成功", f"配置已保存到:\n{DEFAULT_CONFIG_PATH}")
        else:
            QMessageBox.warning(self, "警告", "保存配置失败")
    
    def get_configured_cases(self) -> list:
        """获取已配置的算例列表"""
        return [c for c in self.cases if c.is_configured]
    
    def get_all_cases(self) -> list:
        """获取所有算例（包括未配置的）"""
        return self.cases.copy()
    
    def accept(self):
        """确定按钮"""
        # 检查是否有算例
        if not self.cases:
            QMessageBox.warning(self, "警告", "请至少添加一个算例")
            return
        
        # 检查是否所有算例都已配置
        unconfigured = [c for c in self.cases if not c.is_configured]
        if unconfigured:
            reply = QMessageBox.question(
                self, "未配置的算例",
                f"有 {len(unconfigured)} 个算例尚未配置数据，\n"
                "这些算例将使用自动生成的数据。\n\n继续吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # 自动保存配置
        save_cases_config(self.cases, DEFAULT_CONFIG_PATH)
        
        super().accept()
