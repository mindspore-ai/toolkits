import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QSizePolicy, QHeaderView)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Upper part - input area
        input_group = QGroupBox("输入参数")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)

        # Kernel Details file selection
        kernel_layout = QHBoxLayout()
        self.kernel_btn = QPushButton("选择 Kernel Details 文件")
        self.kernel_btn.clicked.connect(self.select_kernel_file)
        self.kernel_label = QLabel("未选择文件")
        self.kernel_label.setWordWrap(True)
        self.kernel_label.setMinimumHeight(10)
        kernel_layout.addWidget(self.kernel_btn)
        kernel_layout.addWidget(self.kernel_label)
        input_layout.addWidget(QLabel("必选: Kernel Details 文件"))
        input_layout.addLayout(kernel_layout)

        # IR path selection
        ir_layout = QHBoxLayout()
        self.ir_btn = QPushButton("选择 IR 图文件夹 (可选)")
        self.ir_btn.clicked.connect(self.select_ir_folder)
        self.ir_label = QLabel("未选择文件夹 (可选输入)")
        self.ir_label.setWordWrap(True)
        self.ir_label.setMinimumHeight(10)
        ir_layout.addWidget(self.ir_btn)
        ir_layout.addWidget(self.ir_label)
        input_layout.addWidget(QLabel("可选: IR 图路径, 需包含graph_build_*.ir文件(用于重计算分析、allgather精确拆分tp/dp)"))
        input_layout.addLayout(ir_layout)

        # Output file selection
        output_layout = QHBoxLayout()
        self.output_btn = QPushButton("选择性能拆解输出文件")
        self.output_btn.clicked.connect(self.select_output_file)
        self.output_label = QLabel("未选择文件")
        self.output_label.setWordWrap(True)
        self.output_label.setMinimumHeight(10)
        output_layout.addWidget(self.output_btn)
        output_layout.addWidget(self.output_label)
        input_layout.addWidget(QLabel("必选: 性能拆解输出文件"))
        input_layout.addLayout(output_layout)

        # Analysis button
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        input_layout.addWidget(self.analyze_btn)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group, stretch=1)

        # Lower part - result display area
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()

        # Chart display area
        self.figure = Figure(figsize=(12, 9))
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(
            left=0.1, right=0.95,
            bottom=0.1, top=0.9,
            wspace=0.3, hspace=0.3
        )
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout.addWidget(self.canvas, stretch=7)

        # Data table area
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        result_layout.addWidget(self.table, stretch=2)

        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group, stretch=10)

        self.setLayout(main_layout)

        self.kernel_file = ""
        self.ir_folder = ""
        self.output_file = ""

    def select_kernel_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "选择 Kernel Details 文件",
                                              "", "All Files (*);;Text Files (*.txt)",
                                              options=options)
        if file:
            self.kernel_file = file
            self.kernel_label.setText(file)
            self.kernel_label.setToolTip(file)

    def select_ir_folder(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, "选择 IR 图文件夹",
                                                  options=options)
        if folder:
            self.ir_folder = folder
            self.ir_label.setText(folder)
            self.ir_label.setToolTip(folder)

    def select_output_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getSaveFileName(self, "选择性能拆解输出文件",
                                              "", "Excel Files (*.xlsx);;All Files (*)",
                                              options=options)
        if file:
            self.output_file = file
            self.output_label.setText(file)
            self.output_label.setToolTip(file)

    def analyze(self):
        '''analyze'''
        if not self.kernel_file or not self.output_file:
            QMessageBox.warning(self, "输入错误", "必须选择 Kernel Details 文件和输出文件!")
            return

        try:
            # Empty the previous charts.
            self.figure.clear()

            from overlap_analyse import Overlap_Analyzer
            title = "profile analy"
            analyzer = Overlap_Analyzer(self.kernel_file, save_path=self.output_file)
            overlap_statistic = analyzer.analyse(title=title)
            analyzer.analy_recompute(self.ir_folder)
            analyzer.analy_parallel()
            analyzer.us_to_ms()
            analyzer.save_result()

            axs = self.figure.subplots(2, 2)
            analyzer.plot_figure(axs)
            self.figure.tight_layout()
            self.canvas.draw()

            op_analy_result = analyzer.get_op_analy_result()
            self.update_table(op_analy_result)

            QMessageBox.information(self, "分析完成", "分析成功完成并保存结果!")
            analyzer = None
            import gc
            gc.collect()

        except Exception as e:
            QMessageBox.critical(self, "分析错误", f"分析过程中发生错误:\n{str(e)}")

    def update_table(self, data):
        """Table displays data."""
        categories = list(data.keys())
        metrics = list(data[categories[0]].keys()) if categories else []

        self.table.setRowCount(len(categories))
        self.table.setColumnCount(len(metrics) + 1)

        headers = ['类别'] + list(metrics)
        self.table.setHorizontalHeaderLabels(headers)

        for row, category in enumerate(categories):
            item = QTableWidgetItem(category)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item)

            for col, metric in enumerate(metrics, start=1):
                value = data[category].get(metric, '')
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

class MainWindow(QMainWindow):
    '''MainWindow'''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("性能分析工具")
        self.setGeometry(100, 100, 1200, 900)

        try:
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon('icon.png'))
        except:
            pass
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)

        self.add_tab_btn = QPushButton("+ 新建分析")
        self.add_tab_btn.setStyleSheet("padding: 5px;")
        self.add_tab_btn.clicked.connect(self.add_new_tab)
        self.tabs.setCornerWidget(self.add_tab_btn)

        self.add_new_tab()
        self.setCentralWidget(self.tabs)

    def add_new_tab(self):
        tab = AnalysisTab()
        index = self.tabs.addTab(tab, f"分析 {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(index)

    def close_tab(self, index):
        if self.tabs.count() > 1:
            self.tabs.removeTab(index)
        else:
            QMessageBox.information(self, "提示", "至少需要保留一个标签页")

    def closeEvent(self, event):
        self.process.terminate()
        self.process.waitForFinished()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid gray;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QTableWidget {
            font-size: 11px;
        }
        QLabel {
            color: #333;
        }
        QLabel[accessibleName="pathLabel"] {
            font-family: Consolas, monospace;
            background-color: #f5f5f5;
            padding: 2px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
    """)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())