"""
参数输入面板模块
Input Panel Module

提供问题参数和算法参数的输入界面。
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QPushButton, QComboBox,
    QScrollArea, QFrame, QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class InputPanel(QWidget):
    """
    参数输入面板
    
    包含问题参数和算法参数的配置界面。
    """
    
    # 信号: 参数变化时发出
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # 减少间距
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)  # 更紧凑的间距
        
        # 1. 数据输入模式
        self.mode_group = self._create_mode_group()
        scroll_layout.addWidget(self.mode_group)
        
        # 2. 问题规模
        self.problem_group = self._create_problem_group()
        scroll_layout.addWidget(self.problem_group)
        
        # 3. 算法参数 (可折叠)
        self.algorithm_group = self._create_algorithm_group()
        scroll_layout.addWidget(self.algorithm_group)
        
        # 4. 高级设置
        self.advanced_group = self._create_advanced_group()
        scroll_layout.addWidget(self.advanced_group)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def _create_mode_group(self) -> QGroupBox:
        """创建数据输入模式选择组"""
        group = QGroupBox("数据输入模式")
        layout = QVBoxLayout(group)
        
        self.mode_button_group = QButtonGroup(self)
        
        self.auto_mode = QRadioButton("自动生成 (推荐)")
        self.auto_mode.setChecked(True)
        self.auto_mode.setToolTip("系统将自动生成符合逻辑的测试数据")
        
        self.manual_mode = QRadioButton("手动输入")
        self.manual_mode.setToolTip("需要手动输入所有加工时间、能耗等数据")
        
        self.mode_button_group.addButton(self.auto_mode, 0)
        self.mode_button_group.addButton(self.manual_mode, 1)
        
        # 连接模式切换信号
        self.mode_button_group.buttonClicked.connect(self._on_mode_changed)
        
        layout.addWidget(self.auto_mode)
        layout.addWidget(self.manual_mode)
        
        # 模式说明标签
        self.mode_description = QLabel("📊 系统将自动生成符合逻辑的随机测试数据")
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #1976D2; padding: 5px; background: #E3F2FD; border-radius: 4px;")
        layout.addWidget(self.mode_description)
        
        # 手动输入按钮 (默认隐藏)
        self.manual_input_btn = QPushButton("📝 打开数据输入界面")
        self.manual_input_btn.setToolTip("点击输入加工时间、设置时间、能耗等详细数据")
        self.manual_input_btn.clicked.connect(self._open_manual_input_dialog)
        self.manual_input_btn.setVisible(False)
        layout.addWidget(self.manual_input_btn)
        
        # 手动输入状态标签
        self.manual_status_label = QLabel("")
        self.manual_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.manual_status_label.setVisible(False)
        layout.addWidget(self.manual_status_label)
        
        # 随机种子 (仅自动模式显示)
        self.seed_layout_widget = QWidget()
        seed_layout = QHBoxLayout(self.seed_layout_widget)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_label = QLabel("随机种子:")
        seed_label.setToolTip("设置随机种子以获得可重复的结果")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText("随机")
        seed_layout.addWidget(seed_label)
        seed_layout.addWidget(self.seed_spin)
        seed_layout.addStretch()
        
        layout.addWidget(self.seed_layout_widget)
        
        # 存储手动输入的数据
        self.manual_data = None
        
        return group
    
    def _on_mode_changed(self):
        """模式切换时的处理"""
        is_manual = self.manual_mode.isChecked()
        
        if is_manual:
            self.mode_description.setText(
                "📝 手动输入模式: 请点击下方按钮输入每个阶段每台机器的加工时间、设置时间、速度参数和能耗成本"
            )
            self.mode_description.setStyleSheet("color: #FF5722; padding: 5px; background: #FBE9E7; border-radius: 4px;")
            self.manual_input_btn.setVisible(True)
            self.seed_layout_widget.setVisible(False)
            self._update_manual_status()
        else:
            self.mode_description.setText("📊 系统将自动生成符合逻辑的随机测试数据")
            self.mode_description.setStyleSheet("color: #1976D2; padding: 5px; background: #E3F2FD; border-radius: 4px;")
            self.manual_input_btn.setVisible(False)
            self.manual_status_label.setVisible(False)
            self.seed_layout_widget.setVisible(True)
    
    def _open_manual_input_dialog(self):
        """打开手动数据输入对话框"""
        from ui.manual_input_dialog import ManualDataInputDialog
        
        dialog = ManualDataInputDialog(
            n_jobs=self.n_jobs_spin.value(),
            n_stages=self.n_stages_spin.value(),
            machines_per_stage=self.machines_spin.value(),
            n_speed_levels=self.n_speeds_spin.value(),
            n_skill_levels=self.n_skills_spin.value(),
            parent=self
        )
        
        if dialog.exec_() == dialog.Accepted:
            self.manual_data = dialog.get_data()
            self._update_manual_status()
    
    def _update_manual_status(self):
        """更新手动输入状态显示"""
        if self.manual_data is not None:
            self.manual_status_label.setText("✅ 数据已输入完成")
            self.manual_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.manual_status_label.setText("⚠️ 尚未输入数据，请点击上方按钮")
            self.manual_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.manual_status_label.setVisible(True)
    
    def _create_problem_group(self) -> QGroupBox:
        """创建问题规模设置组"""
        group = QGroupBox("问题规模")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        
        # 工件数
        row = 0
        jobs_label = QLabel("工件数量:")
        jobs_label.setToolTip("需要调度的工件(工作)数量")
        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(2, 100)
        self.n_jobs_spin.setValue(10)
        layout.addWidget(jobs_label, row, 0)
        layout.addWidget(self.n_jobs_spin, row, 1)
        
        # 阶段数
        row += 1
        stages_label = QLabel("阶段数量:")
        stages_label.setToolTip("生产过程的阶段数")
        self.n_stages_spin = QSpinBox()
        self.n_stages_spin.setRange(1, 20)
        self.n_stages_spin.setValue(5)
        layout.addWidget(stages_label, row, 0)
        layout.addWidget(self.n_stages_spin, row, 1)
        
        # 每阶段机器数
        row += 1
        machines_label = QLabel("每阶段机器数:")
        machines_label.setToolTip("每个阶段可用的并行机器数量")
        self.machines_spin = QSpinBox()
        self.machines_spin.setRange(1, 10)
        self.machines_spin.setValue(3)
        layout.addWidget(machines_label, row, 0)
        layout.addWidget(self.machines_spin, row, 1)
        
        # 速度等级数
        row += 1
        speed_label = QLabel("速度等级数:")
        speed_label.setToolTip("机器可运行的速度等级数 (如: 低速/中速/高速)")
        self.n_speeds_spin = QSpinBox()
        self.n_speeds_spin.setRange(1, 5)
        self.n_speeds_spin.setValue(3)
        layout.addWidget(speed_label, row, 0)
        layout.addWidget(self.n_speeds_spin, row, 1)
        
        # 技能等级数
        row += 1
        skill_label = QLabel("工人技能等级数:")
        skill_label.setToolTip("工人的技能划分等级数")
        self.n_skills_spin = QSpinBox()
        self.n_skills_spin.setRange(1, 5)
        self.n_skills_spin.setValue(3)
        layout.addWidget(skill_label, row, 0)
        layout.addWidget(self.n_skills_spin, row, 1)
        
        return group
    
    def _create_algorithm_group(self) -> QGroupBox:
        """创建算法参数设置组"""
        group = QGroupBox("算法参数 (可调节)")
        layout = QVBoxLayout(group)
        
        # NSGA-II 参数
        nsga_frame = QFrame()
        nsga_layout = QGridLayout(nsga_frame)
        nsga_layout.setSpacing(8)
        
        title_label = QLabel("NSGA-II 参数")
        title_label.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        nsga_layout.addWidget(title_label, 0, 0, 1, 2)
        
        # 种群大小
        pop_label = QLabel("种群大小:")
        pop_label.setToolTip("每一代的解的数量")
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(10, 200)
        self.pop_size_spin.setValue(50)
        nsga_layout.addWidget(pop_label, 1, 0)
        nsga_layout.addWidget(self.pop_size_spin, 1, 1)
        
        # 进化代数
        gen_label = QLabel("进化代数:")
        gen_label.setToolTip("遗传算法的迭代次数")
        self.n_generations_spin = QSpinBox()
        self.n_generations_spin.setRange(10, 500)
        self.n_generations_spin.setValue(100)
        nsga_layout.addWidget(gen_label, 2, 0)
        nsga_layout.addWidget(self.n_generations_spin, 2, 1)
        
        # 交叉概率
        cross_label = QLabel("交叉概率:")
        self.crossover_spin = QDoubleSpinBox()
        self.crossover_spin.setRange(0.1, 1.0)
        self.crossover_spin.setSingleStep(0.05)
        self.crossover_spin.setValue(0.9)
        nsga_layout.addWidget(cross_label, 3, 0)
        nsga_layout.addWidget(self.crossover_spin, 3, 1)
        
        # 变异概率
        mut_label = QLabel("变异概率:")
        self.mutation_spin = QDoubleSpinBox()
        self.mutation_spin.setRange(0.01, 0.5)
        self.mutation_spin.setSingleStep(0.01)
        self.mutation_spin.setValue(0.1)
        nsga_layout.addWidget(mut_label, 4, 0)
        nsga_layout.addWidget(self.mutation_spin, 4, 1)
        
        layout.addWidget(nsga_frame)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # MOSA 参数
        mosa_frame = QFrame()
        mosa_layout = QGridLayout(mosa_frame)
        mosa_layout.setSpacing(8)
        
        mosa_title = QLabel("MOSA 参数")
        mosa_title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        mosa_layout.addWidget(mosa_title, 0, 0, 1, 2)
        
        # 初始温度
        temp_label = QLabel("初始温度:")
        temp_label.setToolTip("模拟退火的起始温度")
        self.init_temp_spin = QDoubleSpinBox()
        self.init_temp_spin.setRange(10, 1000)
        self.init_temp_spin.setValue(100)
        mosa_layout.addWidget(temp_label, 1, 0)
        mosa_layout.addWidget(self.init_temp_spin, 1, 1)
        
        # 冷却系数
        cool_label = QLabel("冷却系数:")
        cool_label.setToolTip("温度衰减系数 (0 < α < 1)")
        self.cooling_spin = QDoubleSpinBox()
        self.cooling_spin.setRange(0.80, 0.99)
        self.cooling_spin.setSingleStep(0.01)
        self.cooling_spin.setValue(0.95)
        mosa_layout.addWidget(cool_label, 2, 0)
        mosa_layout.addWidget(self.cooling_spin, 2, 1)
        
        # 终止温度
        end_label = QLabel("终止温度:")
        self.end_temp_spin = QDoubleSpinBox()
        self.end_temp_spin.setRange(0.1, 10)
        self.end_temp_spin.setValue(1.0)
        mosa_layout.addWidget(end_label, 3, 0)
        mosa_layout.addWidget(self.end_temp_spin, 3, 1)
        
        # MOSA迭代次数
        mosa_iter_label = QLabel("最大迭代数:")
        self.mosa_iterations_spin = QSpinBox()
        self.mosa_iterations_spin.setRange(10, 200)
        self.mosa_iterations_spin.setValue(50)
        mosa_layout.addWidget(mosa_iter_label, 4, 0)
        mosa_layout.addWidget(self.mosa_iterations_spin, 4, 1)
        
        layout.addWidget(mosa_frame)
        
        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)
        
        # VNS 参数
        vns_frame = QFrame()
        vns_layout = QGridLayout(vns_frame)
        vns_layout.setSpacing(8)
        
        vns_title = QLabel("VNS 参数")
        vns_title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        vns_layout.addWidget(vns_title, 0, 0, 1, 2)
        
        # VNS迭代次数
        vns_iter_label = QLabel("局部搜索迭代:")
        vns_iter_label.setToolTip("每次VNS局部搜索的最大迭代次数")
        self.vns_iterations_spin = QSpinBox()
        self.vns_iterations_spin.setRange(5, 50)
        self.vns_iterations_spin.setValue(10)
        vns_layout.addWidget(vns_iter_label, 1, 0)
        vns_layout.addWidget(self.vns_iterations_spin, 1, 1)
        
        # 邻居数量
        neighbors_label = QLabel("邻居采样数:")
        neighbors_label.setToolTip("每个邻域结构生成的邻居解数量")
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 10)
        self.neighbors_spin.setValue(3)
        vns_layout.addWidget(neighbors_label, 2, 0)
        vns_layout.addWidget(self.neighbors_spin, 2, 1)
        
        layout.addWidget(vns_frame)
        
        return group
    
    def _create_advanced_group(self) -> QGroupBox:
        """创建高级设置组"""
        group = QGroupBox("高级设置")
        layout = QGridLayout(group)
        layout.setSpacing(8)
        
        # 目标权重
        weights_label = QLabel("目标权重 (F1:F2:F3):")
        weights_label.setToolTip("用于VNS/MOSA标量化的目标权重")
        
        self.weight_f1_spin = QDoubleSpinBox()
        self.weight_f1_spin.setRange(0.1, 10)
        self.weight_f1_spin.setValue(1.0)
        self.weight_f1_spin.setSingleStep(0.1)
        
        self.weight_f2_spin = QDoubleSpinBox()
        self.weight_f2_spin.setRange(0.1, 10)
        self.weight_f2_spin.setValue(1.0)
        self.weight_f2_spin.setSingleStep(0.1)
        
        self.weight_f3_spin = QDoubleSpinBox()
        self.weight_f3_spin.setRange(0.1, 10)
        self.weight_f3_spin.setValue(1.0)
        self.weight_f3_spin.setSingleStep(0.1)
        
        layout.addWidget(weights_label, 0, 0)
        
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weight_f1_spin)
        weights_layout.addWidget(QLabel(":"))
        weights_layout.addWidget(self.weight_f2_spin)
        weights_layout.addWidget(QLabel(":"))
        weights_layout.addWidget(self.weight_f3_spin)
        layout.addLayout(weights_layout, 0, 1)
        
        # 代表解数量
        rep_label = QLabel("代表解数量:")
        rep_label.setToolTip("MOSA中用于局部搜索的代表解数量")
        self.n_representative_spin = QSpinBox()
        self.n_representative_spin.setRange(3, 30)
        self.n_representative_spin.setValue(10)
        layout.addWidget(rep_label, 1, 0)
        layout.addWidget(self.n_representative_spin, 1, 1)
        
        return group
    
    def get_parameters(self) -> dict:
        """
        获取所有参数值
        
        Returns:
            参数字典
        """
        return {
            # 数据模式
            'auto_mode': self.auto_mode.isChecked(),
            'seed': self.seed_spin.value() if self.seed_spin.value() > 0 else None,
            'manual_data': self.manual_data,  # 手动输入的数据
            
            # 问题规模
            'n_jobs': self.n_jobs_spin.value(),
            'n_stages': self.n_stages_spin.value(),
            'machines_per_stage': self.machines_spin.value(),
            'n_speed_levels': self.n_speeds_spin.value(),
            'n_skill_levels': self.n_skills_spin.value(),
            
            # NSGA-II参数
            'pop_size': self.pop_size_spin.value(),
            'n_generations': self.n_generations_spin.value(),
            'crossover_prob': self.crossover_spin.value(),
            'mutation_prob': self.mutation_spin.value(),
            
            # MOSA参数
            'initial_temp': self.init_temp_spin.value(),
            'cooling_rate': self.cooling_spin.value(),
            'final_temp': self.end_temp_spin.value(),
            'mosa_iterations': self.mosa_iterations_spin.value(),
            
            # VNS参数
            'vns_iterations': self.vns_iterations_spin.value(),
            'neighbors_per_structure': self.neighbors_spin.value(),
            
            # 高级设置
            'weights': (
                self.weight_f1_spin.value(),
                self.weight_f2_spin.value(),
                self.weight_f3_spin.value()
            ),
            'n_representative': self.n_representative_spin.value()
        }
    
    def set_enabled(self, enabled: bool):
        """设置面板启用/禁用状态"""
        self.mode_group.setEnabled(enabled)
        self.problem_group.setEnabled(enabled)
        self.algorithm_group.setEnabled(enabled)
        self.advanced_group.setEnabled(enabled)
