"""
UI样式模块
UI Styles Module

定义PyQt5界面的科学风格样式表。
"""

# 主题颜色
COLORS = {
    'primary': '#1976D2',           # 主色调 - 深蓝
    'primary_light': '#42A5F5',     # 浅蓝
    'primary_dark': '#0D47A1',      # 深蓝
    'secondary': '#00ACC1',         # 青色
    'accent': '#FF7043',            # 橙色强调
    'background': '#FAFAFA',        # 背景
    'surface': '#FFFFFF',           # 表面
    'error': '#E53935',             # 错误
    'success': '#43A047',           # 成功
    'warning': '#FB8C00',           # 警告
    'text_primary': '#212121',      # 主文字
    'text_secondary': '#757575',    # 次要文字
    'border': '#E0E0E0',            # 边框
    'hover': '#E3F2FD',             # 悬停
}

# 主样式表
MAIN_STYLESHEET = f"""
/* 全局样式 */
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
    font-size: 10pt;
    color: {COLORS['text_primary']};
}}

/* 分组框 */
QGroupBox {{
    font-weight: bold;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    background-color: {COLORS['surface']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['primary']};
}}

/* 标签 */
QLabel {{
    color: {COLORS['text_primary']};
}}

/* 输入框 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    padding: 6px 10px;
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['surface']};
    min-height: 24px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 20px;
    border: none;
}}

/* 下拉框 */
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {COLORS['text_secondary']};
}}

/* 按钮 */
QPushButton {{
    padding: 8px 20px;
    border: none;
    border-radius: 4px;
    background-color: {COLORS['primary']};
    color: white;
    font-weight: bold;
    min-height: 32px;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton:disabled {{
    background-color: #BDBDBD;
    color: #FAFAFA;
}}

/* 次要按钮 */
QPushButton.secondary {{
    background-color: {COLORS['surface']};
    color: {COLORS['primary']};
    border: 1px solid {COLORS['primary']};
}}

QPushButton.secondary:hover {{
    background-color: {COLORS['hover']};
}}

/* 成功按钮 */
QPushButton.success {{
    background-color: {COLORS['success']};
}}

/* 单选按钮 */
QRadioButton {{
    spacing: 8px;
}}

QRadioButton::indicator {{
    width: 18px;
    height: 18px;
}}

/* 复选框 */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
}}

/* 进度条 */
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: #E3F2FD;
    text-align: center;
    min-height: 20px;
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 4px;
}}

/* 选项卡 */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['surface']};
}}

QTabBar::tab {{
    padding: 8px 16px;
    margin-right: 4px;
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    background-color: #ECEFF1;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['surface']};
    border-bottom: 2px solid {COLORS['primary']};
}}

QTabBar::tab:hover {{
    background-color: {COLORS['hover']};
}}

/* 文本框 */
QTextEdit, QPlainTextEdit {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['surface']};
    padding: 8px;
}}

/* 滚动条 */
QScrollBar:vertical {{
    border: none;
    background-color: #F5F5F5;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: #BDBDBD;
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: #9E9E9E;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* 表格 */
QTableWidget {{
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    background-color: {COLORS['surface']};
}}

QTableWidget::item {{
    padding: 4px;
}}

QTableWidget::item:selected {{
    background-color: {COLORS['hover']};
    color: {COLORS['text_primary']};
}}

QHeaderView::section {{
    background-color: #ECEFF1;
    padding: 6px;
    border: none;
    border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    font-weight: bold;
}}

/* 状态栏 */
QStatusBar {{
    background-color: #ECEFF1;
    border-top: 1px solid {COLORS['border']};
}}

/* 工具提示 */
QToolTip {{
    background-color: {COLORS['primary_dark']};
    color: white;
    border: none;
    padding: 6px;
    border-radius: 4px;
}}

/* 滑块 */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background-color: #E0E0E0;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['primary']};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {COLORS['primary_light']};
}}
"""

# 运行按钮特殊样式
RUN_BUTTON_STYLE = f"""
QPushButton {{
    font-size: 12pt;
    padding: 12px 32px;
    min-height: 40px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS['primary']}, 
                                stop:1 {COLORS['secondary']});
}}

QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {COLORS['primary_light']}, 
                                stop:1 #26C6DA);
}}
"""

# 结果面板样式
RESULT_PANEL_STYLE = f"""
QFrame {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}
"""
