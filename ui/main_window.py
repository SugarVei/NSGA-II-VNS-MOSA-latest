"""
主窗口模块
Main Window Module

整合所有UI组件，提供完整的用户界面。
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QProgressBar, QLabel,
    QStatusBar, QMessageBox, QFrame, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import sys
import os
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.styles import MAIN_STYLESHEET, RUN_BUTTON_STYLE
from ui.input_panel import InputPanel
from ui.result_panel import ResultPanel

from models.problem import SchedulingProblem
from models.solution import Solution
from models.decoder import Decoder
from algorithms.nsga2 import NSGAII
from algorithms.vns import VNS
from algorithms.mosa import MOSA


class OptimizationWorker(QThread):
    """
    优化算法工作线程
    
    在后台运行优化算法，避免阻塞UI。
    """
    
    # 信号
    progress = pyqtSignal(int, int, str)  # current, total, message
    log = pyqtSignal(str)  # 日志消息
    nsga2_finished = pyqtSignal(list, dict)  # pareto解, 收敛数据
    mosa_finished = pyqtSignal(list, dict)  # pareto解, 收敛数据
    error = pyqtSignal(str)  # 错误消息
    finished = pyqtSignal()  # 完成信号
    
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._is_cancelled = False
    
    def run(self):
        """运行优化"""
        try:
            params = self.params
            
            # 1. 创建问题实例
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 创建调度问题...")
            self.progress.emit(0, 100, "创建问题实例...")
            
            if params['auto_mode']:
                machines_per_stage = [params['machines_per_stage']] * params['n_stages']
                problem = SchedulingProblem.generate_random(
                    n_jobs=params['n_jobs'],
                    n_stages=params['n_stages'],
                    machines_per_stage=machines_per_stage,
                    n_speed_levels=params['n_speed_levels'],
                    n_skill_levels=params['n_skill_levels'],
                    seed=params['seed']
                )
            else:
                # 手动输入模式: 使用用户输入的数据
                manual_data = params.get('manual_data')
                machines_per_stage = [params['machines_per_stage']] * params['n_stages']
                
                if manual_data is not None:
                    self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 使用手动输入的数据...")
                    
                    # 创建问题实例并设置手动输入的数据
                    import numpy as np
                    
                    # 手动输入的加工时间需要扩展到包含速度维度
                    # manual_data['processing_time'] 是 [job, stage, machine]
                    # 需要转换为 [job, stage, machine, speed]
                    base_proc_time = manual_data['processing_time']
                    n_jobs, n_stages, n_machines = base_proc_time.shape
                    n_speeds = params['n_speed_levels']
                    
                    # 扩展加工时间到速度维度 (高速更快)
                    processing_time = np.zeros((n_jobs, n_stages, n_machines, n_speeds))
                    for job in range(n_jobs):
                        for stage in range(n_stages):
                            for machine in range(n_machines):
                                base_time = base_proc_time[job, stage, machine]
                                for speed in range(n_speeds):
                                    # 速度越高，时间越短
                                    speed_factor = 1.0 - 0.25 * speed
                                    processing_time[job, stage, machine, speed] = base_time * speed_factor
                    
                    # 使用手动输入的能耗率
                    energy_rate = manual_data['energy_rate']
                    
                    # 使用手动输入的工人工资
                    skill_wages = manual_data['skill_wages']
                    workers_available = manual_data['workers_available']
                    
                    # 技能兼容性: 技能等级i可操作速度0~i
                    skill_compatibility = np.array([i for i in range(params['n_skill_levels'])])
                    
                    problem = SchedulingProblem(
                        n_jobs=params['n_jobs'],
                        n_stages=params['n_stages'],
                        machines_per_stage=machines_per_stage,
                        n_speed_levels=params['n_speed_levels'],
                        n_skill_levels=params['n_skill_levels'],
                        processing_time=processing_time,
                        energy_rate=energy_rate,
                        skill_wages=skill_wages,
                        skill_compatibility=skill_compatibility,
                        workers_available=workers_available
                    )
                else:
                    # 手动模式但未输入数据，使用随机生成
                    self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] 警告: 手动模式但未输入数据，使用随机生成...")
                    problem = SchedulingProblem.generate_random(
                        n_jobs=params['n_jobs'],
                        n_stages=params['n_stages'],
                        machines_per_stage=machines_per_stage,
                        n_speed_levels=params['n_speed_levels'],
                        n_skill_levels=params['n_skill_levels'],
                        seed=params['seed']
                    )
            
            self.log.emit(problem.summary())
            
            if self._is_cancelled:
                return
            
            # 2. 运行NSGA-II
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] 启动NSGA-II算法...")
            
            nsga2 = NSGAII(
                problem=problem,
                pop_size=params['pop_size'],
                n_generations=params['n_generations'],
                crossover_prob=params['crossover_prob'],
                mutation_prob=params['mutation_prob'],
                seed=params['seed']
            )
            
            def nsga2_callback(current, total, msg):
                if self._is_cancelled:
                    return
                # NSGA-II占总进度的50%
                progress = int(current / total * 50)
                self.progress.emit(progress, 100, msg)
                if current % 10 == 0:
                    self.log.emit(f"  {msg}")
            
            nsga2.set_progress_callback(nsga2_callback)
            
            pareto_nsga2 = nsga2.run()
            
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] NSGA-II完成，找到{len(pareto_nsga2)}个Pareto解")
            self.nsga2_finished.emit(pareto_nsga2, nsga2.get_convergence_data())
            
            if self._is_cancelled:
                return
            
            # 3. 运行MOSA
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] 启动MOSA算法...")
            
            mosa = MOSA(
                problem=problem,
                initial_temp=params['initial_temp'],
                cooling_rate=params['cooling_rate'],
                final_temp=params['final_temp'],
                max_iterations=params['mosa_iterations'],
                vns_iterations=params['vns_iterations'],
                n_representative=params['n_representative'],
                weights=params['weights'],
                seed=params['seed']
            )
            
            def mosa_callback(current, total, msg):
                if self._is_cancelled:
                    return
                # MOSA占总进度的50-100%
                progress = 50 + int(current / total * 50)
                self.progress.emit(progress, 100, msg)
                if current % 5 == 0:
                    self.log.emit(f"  {msg}")
            
            mosa.set_progress_callback(mosa_callback)
            
            pareto_mosa = mosa.run(initial_archive=pareto_nsga2)
            
            self.log.emit(f"[{datetime.now().strftime('%H:%M:%S')}] MOSA完成，最终Pareto解数量: {len(pareto_mosa)}")
            self.mosa_finished.emit(pareto_mosa, mosa.get_convergence_data())
            
            self.progress.emit(100, 100, "优化完成！")
            self.log.emit(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 优化流程完成!")
            
        except Exception as e:
            import traceback
            self.error.emit(f"优化过程出错: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()
    
    def cancel(self):
        """取消优化"""
        self._is_cancelled = True


class MainWindow(QMainWindow):
    """
    主窗口
    
    整合参数输入、优化运行和结果展示。
    """
    
    def __init__(self):
        super().__init__()
        
        self.worker: OptimizationWorker = None
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        """初始化UI"""
        self.setWindowTitle("多目标调度优化系统 v1.0")
        self.setMinimumSize(1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧: 输入面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.input_panel = InputPanel()
        left_layout.addWidget(self.input_panel)
        
        # 运行按钮
        self.run_button = QPushButton("🚀 运行优化")
        self.run_button.setStyleSheet(RUN_BUTTON_STYLE)
        self.run_button.clicked.connect(self.start_optimization)
        left_layout.addWidget(self.run_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("⏹ 取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_optimization)
        left_layout.addWidget(self.cancel_button)
        
        left_widget.setMaximumWidth(380)
        left_widget.setMinimumWidth(350)
        splitter.addWidget(left_widget)
        
        # 右侧: 结果面板
        self.result_panel = ResultPanel()
        splitter.addWidget(self.result_panel)
        
        # 设置分割比例 - 左侧更小，右侧更大
        splitter.setSizes([360, 840])
        
        main_layout.addWidget(splitter)
        
        # 进度条
        progress_layout = QHBoxLayout()
        
        self.progress_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(progress_layout)
        
        # 状态栏
        self.statusBar().showMessage("欢迎使用多目标调度优化系统")
    
    def apply_styles(self):
        """应用样式"""
        self.setStyleSheet(MAIN_STYLESHEET)
    
    def start_optimization(self):
        """开始优化"""
        # 获取参数
        params = self.input_panel.get_parameters()
        
        # 验证参数
        if params['n_jobs'] < 2:
            QMessageBox.warning(self, "参数错误", "工件数量至少为2")
            return
        
        # 检查手动输入模式是否已输入数据
        if not params['auto_mode'] and params.get('manual_data') is None:
            reply = QMessageBox.question(
                self, "数据未输入",
                "手动输入模式下尚未输入数据。\n\n是否继续使用随机生成的数据？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # 清空之前的结果
        self.result_panel.clear()
        
        # 禁用输入
        self.input_panel.set_enabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # 创建并启动工作线程
        self.worker = OptimizationWorker(params)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.result_panel.append_log)
        self.worker.nsga2_finished.connect(self.on_nsga2_finished)
        self.worker.mosa_finished.connect(self.on_mosa_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        
        self.result_panel.append_log(f"[{datetime.now().strftime('%H:%M:%S')}] 开始优化...")
        self.result_panel.append_log(f"参数: 工件={params['n_jobs']}, 阶段={params['n_stages']}, 机器={params['machines_per_stage']}")
        
        self.worker.start()
    
    def cancel_optimization(self):
        """取消优化"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.result_panel.append_log("⚠️ 正在取消优化...")
            self.statusBar().showMessage("正在取消...")
    
    def on_progress(self, current: int, total: int, message: str):
        """进度更新"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_nsga2_finished(self, pareto_solutions: list, convergence_data: dict):
        """NSGA-II完成"""
        self.convergence_data = {'NSGA-II': convergence_data}
        self.statusBar().showMessage(f"NSGA-II完成，找到{len(pareto_solutions)}个Pareto解")
    
    def on_mosa_finished(self, pareto_solutions: list, convergence_data: dict):
        """MOSA完成"""
        self.convergence_data['MOSA'] = convergence_data
        
        # 更新结果面板
        self.result_panel.update_pareto_solutions(pareto_solutions, "MOSA")
        self.result_panel.update_convergence(self.convergence_data)
        
        self.statusBar().showMessage(f"优化完成，找到{len(pareto_solutions)}个Pareto解")
    
    def on_error(self, error_message: str):
        """错误处理"""
        QMessageBox.critical(self, "优化错误", error_message)
        self.result_panel.append_log(f"❌ 错误: {error_message}")
    
    def on_finished(self):
        """优化完成"""
        self.input_panel.set_enabled(True)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        if self.progress_bar.value() >= 100:
            self.statusBar().showMessage("✅ 优化完成")
        else:
            self.statusBar().showMessage("⚠️ 优化已取消")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "优化正在运行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """程序入口"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
