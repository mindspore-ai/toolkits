import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QSizePolicy, QHeaderView, QProgressBar,
                             QGridLayout, QLineEdit)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from di_analyzer import Analyzer
from utils.static_value import PARALLEL_TP, PARALLEL_DP, PARALLEL_CP, PARALLEL_PP, PARALLEL_EP

class AnalysisWorker(QObject):
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    progress_text_updated = pyqtSignal(str)

    def __init__(self, kernel_file, parallel_config, ir_folder, output_file):
        super().__init__()
        self.kernel_file = kernel_file
        self.parallel_config = parallel_config
        self.ir_folder = ir_folder
        self.output_file = output_file
        self.analyzer = None

    def run_analysis(self):
        try:
            total_steps = 8
            self.progress_text_updated.emit("初始化...")
            self.progress_updated.emit(int(1 / total_steps * 100))
            start_time = time.time()
            self.analyzer = Analyzer(self.kernel_file, self.parallel_config, self.ir_folder, save_path=self.output_file)
            print(f"1. Analyzer init time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("通信掩盖分析中...")
            self.progress_updated.emit(int(2 / total_steps * 100))
            start_time = time.time()
            self.analyzer.analy_overlap()
            print(f"2. overlap analyse time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("重计算分析中...")
            self.progress_updated.emit(int(3 / total_steps * 100))
            start_time = time.time()
            self.analyzer.update_recompute_summary()
            print(f"3. recompute analyse time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("并行分析中...")
            self.progress_updated.emit(int(4 / total_steps * 100))
            start_time = time.time()
            self.analyzer.analy_parallel()
            print(f"4. analy_parallel time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("vector分析中...")
            self.progress_updated.emit(int(5 / total_steps * 100))
            start_time = time.time()
            self.analyzer.analy_vector()
            print(f"5. analy_vector time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("数据格式化中...")
            self.progress_updated.emit(int(6 / total_steps * 100))
            start_time = time.time()
            self.analyzer.format_result()
            print(f"6. us_to_ms time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("分析已完成，数据保存中...")
            self.progress_updated.emit(int(7 / total_steps * 100))
            start_time = time.time()
            self.analyzer.save_result()
            print(f"7. save_result time = {time.time() - start_time : .2f}.")
            self.progress_text_updated.emit("界面图表显示刷新中...")
            self.progress_updated.emit(int(8 / total_steps * 100))
            start_time = time.time()
            op_analy_result = self.analyzer.get_op_analy_result()
            print(f"8. op_analy_result time = {time.time() - start_time : .2f}.")
            self.finished.emit(op_analy_result)
        except Exception as e:
            self.error.emit(f"分析过程中发生错误:\n{str(e)}")

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker_thread = None
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)
        self.analysis_start_time = 0
        self.current_step_text = ""

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        top_area_layout = QHBoxLayout()
        input_group = QGroupBox("输入参数")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(10)

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

        ir_layout = QHBoxLayout()
        self.ir_btn = QPushButton("选择 IR 图文件夹 (可选)")
        self.ir_btn.clicked.connect(self.select_ir_folder)
        self.ir_label = QLabel("未选择文件夹 (可选输入)")
        self.ir_label.setWordWrap(True)
        self.ir_label.setMinimumHeight(10)
        ir_layout.addWidget(self.ir_btn)
        ir_layout.addWidget(self.ir_label)
        input_layout.addWidget(QLabel("可选: IR 图路径, 需包含graph_build_*.ir文件(用于重计算分析、精确区分tp/dp/cp)"))
        input_layout.addLayout(ir_layout)

        parallel_group = QGroupBox("并行配置 (可选)")
        parallel_layout = QGridLayout()
        parallel_labels = [PARALLEL_DP, PARALLEL_CP, PARALLEL_TP, PARALLEL_EP, PARALLEL_PP]
        self.parallel_edits = []
        for i, label in enumerate(parallel_labels):
            lbl = QLabel(label)
            edit = QLineEdit("")
            edit.setValidator(QIntValidator(1, 100))
            edit.setFixedWidth(50)
            lbl.setFixedWidth(25)
            cell_layout = QHBoxLayout()
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(3)
            cell_layout.addWidget(lbl)
            cell_layout.addWidget(edit)
            cell_widget = QWidget()
            cell_widget.setLayout(cell_layout)
            parallel_layout.addWidget(cell_widget, i // 5, i % 5)
            self.parallel_edits.append(edit)
        input_layout.addLayout(parallel_layout)

        output_layout = QHBoxLayout()
        self.output_btn = QPushButton("选择性能拆解输出目录")
        self.output_btn.clicked.connect(self.select_output_file)
        self.output_label = QLabel("未选择文件")
        self.output_label.setWordWrap(True)
        self.output_label.setMinimumHeight(10)
        output_layout.addWidget(self.output_btn)
        output_layout.addWidget(self.output_label)
        input_layout.addWidget(QLabel("可选: 性能拆解输出目录"))
        input_layout.addLayout(output_layout)

        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.analyze)
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        input_layout.addWidget(self.analyze_btn)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)
        input_group.setLayout(input_layout)

        logo_label = QLabel(self)
        logo_label.setFixedSize(256, 256)

        pixmap = QPixmap("di_icon.png")
        logo_label.setPixmap(pixmap)
        logo_label.setScaledContents(True)

        top_area_layout.addWidget(input_group, 1)
        top_area_layout.addWidget(logo_label)
        main_layout.addLayout(top_area_layout)

        top_widget = QWidget()
        top_widget.setLayout(top_area_layout)
        main_layout.addWidget(top_widget)

        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()

        self.figure = Figure(figsize=(12, 9))
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(
            left=0.1, right=0.95,
            bottom=0.1, top=0.9,
            wspace=0.3, hspace=0.3
        )
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout.addWidget(self.canvas, stretch=7)

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
        self.parallel_config = {PARALLEL_TP: None, PARALLEL_PP: None, PARALLEL_EP: None, PARALLEL_CP: None, PARALLEL_DP: None}

    def select_kernel_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "选择 Kernel Details 文件",
                                              "", "All Files (*);;Text Files (*.txt)",
                                              options=options)
        if file:
            self.kernel_file = os.path.normpath(file)
            self.kernel_label.setText(self.kernel_file)
            self.kernel_label.setToolTip(self.kernel_file)

    def select_ir_folder(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, "选择 IR 图文件夹",
                                                  options=options)
        if folder:
            self.ir_folder = folder
            self.ir_label.setText(folder)
            self.ir_label.setToolTip(folder)

    def get_parallel_config(self):
        labels = [PARALLEL_DP, PARALLEL_CP, PARALLEL_TP, PARALLEL_EP, PARALLEL_PP]
        for i, edit in enumerate(self.parallel_edits):
            value = edit.text().strip()
            self.parallel_config[labels[i]] = int(value) if (value != None and value != "") else None
        return self.parallel_config

    def select_output_file(self):
        options = QFileDialog.Options()
        file = QFileDialog.getExistingDirectory(
            parent=None,
            caption="选择文件夹",
            directory="",
            options=QFileDialog.ShowDirsOnly
        )
        if file:
            self.output_file = file
            self.output_label.setText(file)
            self.output_label.setToolTip(file)

    def is_input_parallel_config(self):
        return self.parallel_config[PARALLEL_CP] is None or self.parallel_config[PARALLEL_TP] is None or \
               self.parallel_config[PARALLEL_EP] is None or self.parallel_config[PARALLEL_PP] is None or \
               self.parallel_config[PARALLEL_DP] is None

    def analyze(self):
        if not self.kernel_file:
            QMessageBox.warning(self, "输入错误", "必须选择 Kernel Details文件!")
            return

        if self.output_file:
            final_output_path = self.output_file
        else:
            input_directory = os.path.dirname(self.kernel_file)
            final_output_path = input_directory
            self.output_file = final_output_path

        if self.ir_folder:
            self.get_parallel_config()

        if self.is_input_parallel_config() and self.ir_folder:
            QMessageBox.warning(self, "输入错误", "IR图和并行策略必须同时输入!")
            return

        self.output_label.setText(f"输出至: {final_output_path}")
        self.output_label.setToolTip(final_output_path)

        self.analyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.figure.clear()
        self.canvas.draw()
        
        self.worker_thread = QThread()
        self.worker = AnalysisWorker(self.kernel_file, self.parallel_config, self.ir_folder, self.output_file)
        
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.progress_text_updated.connect(self.update_button_text)

        self.worker_thread.started.connect(self.worker.run_analysis)
        self.worker.finished.connect(self.handle_analysis_finished)
        self.worker.error.connect(self.handle_analysis_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        self.analysis_start_time = time.time()
        self.current_step_text = "准备开始..."
        self.timer.start(1000)
        
        self.worker_thread.start()

    def handle_analysis_finished(self, op_analy_result):
        self.timer.stop()
        total_duration_seconds = 0
        if self.analysis_start_time > 0:
            total_duration_seconds = time.time() - self.analysis_start_time
        
        minutes = int(total_duration_seconds // 60)
        seconds = int(total_duration_seconds % 60)
        duration_str = f"{minutes}分 {seconds}秒"

        try:
            axs = self.figure.subplots(2, 2)
            self.worker.analyzer.plot_figure(axs)
            self.figure.tight_layout()
            self.canvas.draw()
            self.update_table(op_analy_result)

            QMessageBox.information(self, f"分析完成 ({duration_str})", "分析成功完成并保存结果!")
        finally:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText("开始分析")
            self.progress_bar.setVisible(False)
            self.worker_thread = None
            self.worker = None

    def handle_analysis_error(self, error_message):
        QMessageBox.critical(self, "分析错误", error_message)
        self.timer.stop()
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("开始分析")
        self.progress_bar.setVisible(False)
        self.worker_thread = None
        self.worker = None

    def update_table(self, data):
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

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_button_text(self, text):
        self.current_step_text = text
        self.update_timer_display()
    
    def update_timer_display(self):
        if self.analysis_start_time > 0:
            elapsed_seconds = time.time() - self.analysis_start_time
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            display_text = f"{self.current_step_text} ({time_str})"
            self.analyze_btn.setText(display_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepInsight - Profile Analysis Tool")
        self.setGeometry(100, 100, 1200, 900)
        self.showMaximized()

        try:
            self.setWindowIcon(QIcon("di_icon.png"))
        except:
            print(f"警告：在当前工作目录下找不到图标文件 'di_icon.png'")
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
        pass
        tab = AnalysisTab()
        index = self.tabs.addTab(tab, f"分析 {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(index)

    def close_tab(self, index):
        if self.tabs.count() > 1:
            self.tabs.removeTab(index)
        else:
            QMessageBox.information(self, "提示", "至少需要保留一个标签页")

    def closeEvent(self, event):
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
        QTabBar::tab {
            background-color: #E0E0E0;
            color: #333;
            border: 1px solid #C0C0C0;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 240px;
            padding: 6px 12px;
        }
        QTabBar::tab:hover {
            background-color: #F0F0F0;
        }
        QTabBar::tab:selected {
            background-color: white;
            color: black;
            font-weight: bold;
            border: 1px solid #C0C0C0;
            border-bottom-color: white;
        }
        QTabWidget::pane {
            border-top: 1px solid #C0C0C0;
        }
    """)
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())