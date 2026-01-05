"""
多目标调度优化系统
Multi-Objective Scheduling Optimization System

主程序入口
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import main

if __name__ == "__main__":
    main()
