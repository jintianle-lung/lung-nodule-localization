import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import serial
import serial.tools.list_ports
import threading
import time
import queue
from collections import deque
import struct
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置系统编码
import sys
import locale
if sys.platform.startswith('win'):
    # Windows系统编码设置
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese (Simplified)_China.936')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
        except:
            pass

# 导入检测器和解析器
import sys
import os
import torch

# 当前目录已包含所有模块
from fusion_real_time_detection import FastProtocolParser, EnhancedNoduleDetectionSystem
from enhanced_stress_detection_system import EnhancedStressNoduleDetectionSystem
from realtime_detector import RealtimeNoduleDetector
from final_model import DualStreamModel
from sequence_dataset import NoduleSequenceDataset

class OptimizedDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Princess & Pea (豌豆公主) 智能医疗检测系统 (Release v1.0)")
        self.root.geometry("1400x900")
        
        # ... (字体设置等保持不变)
        
        # 数据相关
        self.data = None
        self.current_frame = 0
        
        # ... (串口参数保持不变)
        
        # 初始化检测器和解析器
        self.detector = EnhancedNoduleDetectionSystem()
        self.parser = FastProtocolParser()
        
        # 初始化增强的应力传感器检测系统
        self.enhanced_detector = EnhancedStressNoduleDetectionSystem()
        self.use_enhanced_detection = False  # 默认使用原有检测器
        
        
        # 深度学习检测器 (DualStreamModel)
        self.dl_model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
        self.dl_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 加载新模型 (DualStreamModel - Active Learning Optimized)
            self.seq_len = 10
            self.dl_model = DualStreamModel(seq_len=self.seq_len).to(self.device)
            self.dl_model.load_state_dict(torch.load(self.dl_model_path, map_location=self.device))
            self.dl_model.eval()
            print(f"Deep Learning Model (DualStream) loaded successfully from {self.dl_model_path}")
            self.use_dl_detection = True
        except Exception as e:
            print(f"Error loading DL model: {e}")
            # messagebox.showwarning("警告", f"无法加载深度学习模型: {e}\n智能检测功能将受限。")
            self.use_dl_detection = False
            
        self.dl_probability = 0.0
        self.dl_pred_size = 0.0
        self.dl_pred_depth = 0.0
        
        # 帧缓冲区 (用于时序模型)
        from collections import deque
        self.frame_buffer = deque(maxlen=self.seq_len)
        
        # ... (后续初始化保持不变)
            
        self.dl_probability = 0.0
        
        # 结节参数设置
        self.nodule_sphericity = 0.8  # 结节原形度
        self.nodule_size = 10.0  # 结节尺寸(mm)
        
        # 性能监控
        self.fps_counter = deque(maxlen=30)
        self.last_update_time = time.time()
        
        # 应力阈值设置 - 范围控制
        self.stress_min_threshold = 0.0  # 最小阈值
        self.stress_max_threshold = 100.0  # 最大阈值
        self.enable_threshold = True  # 默认启用阈值过滤
        
        # 创建界面
        self.create_widgets()
        
        # 设置窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 启动性能监控
        self.start_performance_monitor()
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 文件控制
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="加载CSV文件", command=self.load_csv_file).pack(side=tk.LEFT, padx=(0, 5))
        
        # 帧控制
        frame_control = ttk.Frame(file_frame)
        frame_control.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        ttk.Label(frame_control, text="帧:").pack(side=tk.LEFT)
        self.frame_var = tk.IntVar()
        
        # 导航按钮区
        nav_frame = ttk.Frame(frame_control)
        nav_frame.pack(side=tk.LEFT, padx=(5, 5))
        
        # 快退/上一帧
        ttk.Button(nav_frame, text="<<", width=3, command=lambda: self.skip_frames_control(-10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="<", width=3, command=self.prev_frame).pack(side=tk.LEFT, padx=1)
        
        # 滑动条
        self.frame_scale = ttk.Scale(frame_control, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.frame_var, command=self.on_frame_change)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        # 下一帧/快进
        next_nav_frame = ttk.Frame(frame_control)
        next_nav_frame.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(next_nav_frame, text=">", width=3, command=self.next_frame).pack(side=tk.LEFT, padx=1)
        ttk.Button(next_nav_frame, text=">>", width=3, command=lambda: self.skip_frames_control(10)).pack(side=tk.LEFT, padx=1)
        
        # 跳转输入框
        jump_frame = ttk.Frame(frame_control)
        jump_frame.pack(side=tk.LEFT, padx=(10, 5))
        ttk.Label(jump_frame, text="跳转:").pack(side=tk.LEFT)
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(jump_frame, textvariable=self.jump_var, width=5)
        jump_entry.pack(side=tk.LEFT, padx=2)
        jump_entry.bind('<Return>', self.jump_to_frame)
        ttk.Button(jump_frame, text="Go", width=3, command=self.jump_to_frame).pack(side=tk.LEFT)
        
        self.frame_label = ttk.Label(frame_control, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # 绑定键盘快捷键
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Control-Left>', lambda e: self.skip_frames_control(-10))
        self.root.bind('<Control-Right>', lambda e: self.skip_frames_control(10))
        
        # 串口控制
        serial_frame = ttk.Frame(control_frame)
        serial_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(serial_frame, text="串口:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=15)
        self.port_combo.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Button(serial_frame, text="刷新端口", command=self.refresh_ports).pack(side=tk.LEFT, padx=(0, 5))
        
        self.serial_btn = ttk.Button(serial_frame, text="连接串口", command=self.toggle_serial)
        self.serial_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 应力阈值控制 - 范围设置
        threshold_frame = ttk.Frame(serial_frame)
        threshold_frame.pack(side=tk.LEFT, padx=(15, 0))
        
        # 阈值启用复选框 - 默认启用
        self.threshold_enable_var = tk.BooleanVar(value=True)
        self.threshold_checkbox = ttk.Checkbutton(threshold_frame, text="阈值范围", 
                                                variable=self.threshold_enable_var,
                                                command=self.toggle_threshold)
        self.threshold_checkbox.pack(side=tk.LEFT, padx=(0, 5))
        
        # 最小阈值设置
        ttk.Label(threshold_frame, text="最小:").pack(side=tk.LEFT, padx=(5, 2))
        self.min_threshold_var = tk.DoubleVar(value=0.0)
        self.min_threshold_spinbox = ttk.Spinbox(threshold_frame, from_=0.0, to=99.0, 
                                               increment=0.1, width=6,
                                               textvariable=self.min_threshold_var,
                                               command=self.update_min_threshold)
        self.min_threshold_spinbox.pack(side=tk.LEFT, padx=(0, 3))
        
        # 最大阈值设置
        ttk.Label(threshold_frame, text="最大:").pack(side=tk.LEFT, padx=(3, 2))
        self.max_threshold_var = tk.DoubleVar(value=100.0)
        self.max_threshold_spinbox = ttk.Spinbox(threshold_frame, from_=1.0, to=1000.0, 
                                               increment=0.1, width=6,
                                               textvariable=self.max_threshold_var,
                                               command=self.update_max_threshold)
        self.max_threshold_spinbox.pack(side=tk.LEFT, padx=(0, 5))
        
        # 阈值状态显示
        self.threshold_status_var = tk.StringVar(value="已启用 (0.0-100.0)")
        self.threshold_status_label = ttk.Label(threshold_frame, textvariable=self.threshold_status_var,
                                              foreground="green")
        self.threshold_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # 检测模式控制
        detection_frame = ttk.Frame(serial_frame)
        detection_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(detection_frame, text="检测模式:").pack(side=tk.LEFT)
        self.detection_mode_var = tk.StringVar(value="标准")
        detection_combo = ttk.Combobox(detection_frame, textvariable=self.detection_mode_var, 
                                     values=["标准", "增强应力"], width=12, state="readonly")
        detection_combo.pack(side=tk.LEFT, padx=(5, 5))
        detection_combo.bind('<<ComboboxSelected>>', self.on_detection_mode_change)
        
        # 训练数据控制
        training_frame = ttk.Frame(serial_frame)
        training_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(training_frame, text="添加训练数据", command=self.show_training_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(training_frame, text="训练系统", command=self.train_enhanced_system).pack(side=tk.LEFT)
        
        # 结节参数控制
        nodule_frame = ttk.LabelFrame(control_frame, text="结节参数设置", padding=5)
        nodule_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 结节原形度设置
        sphericity_frame = ttk.Frame(nodule_frame)
        sphericity_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(sphericity_frame, text="结节原形度:").pack(side=tk.LEFT)
        self.sphericity_var = tk.DoubleVar(value=0.8)
        sphericity_spin = ttk.Spinbox(sphericity_frame, from_=0.1, to=1.0, increment=0.05, 
                                    textvariable=self.sphericity_var, width=8,
                                    command=self.update_nodule_sphericity)
        sphericity_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 结节尺寸设置
        size_frame = ttk.Frame(nodule_frame)
        size_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(size_frame, text="结节尺寸(mm):").pack(side=tk.LEFT)
        self.size_var = tk.DoubleVar(value=10.0)
        size_spin = ttk.Spinbox(size_frame, from_=1.0, to=50.0, increment=0.5, 
                              textvariable=self.size_var, width=8,
                              command=self.update_nodule_size)
        size_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 结节深度设置
        depth_frame = ttk.Frame(nodule_frame)
        depth_frame.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(depth_frame, text="结节深度(mm):").pack(side=tk.LEFT)
        self.depth_var = tk.DoubleVar(value=10.0)
        depth_spin = ttk.Spinbox(depth_frame, from_=1.0, to=50.0, increment=0.5, 
                               textvariable=self.depth_var, width=8)
        depth_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 弹性成像模式
        elastography_frame = ttk.Frame(nodule_frame)
        elastography_frame.pack(side=tk.LEFT)
        
        self.elastography_var = tk.BooleanVar(value=True)
        elastography_check = ttk.Checkbutton(elastography_frame, text="启用弹性成像分析", 
                                           variable=self.elastography_var,
                                           command=self.toggle_elastography)
        elastography_check.pack(side=tk.LEFT)
        
        # 深度学习开关
        dl_frame = ttk.Frame(nodule_frame)
        dl_frame.pack(side=tk.LEFT, padx=10)
        self.dl_var = tk.BooleanVar(value=True)
        dl_check = ttk.Checkbutton(dl_frame, text="启用AI关键帧检测",
                                   variable=self.dl_var)
        dl_check.pack(side=tk.LEFT)
        
        # 性能参数控制
        perf_frame = ttk.Frame(control_frame)
        perf_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(perf_frame, text="更新间隔(ms):").pack(side=tk.LEFT)
        self.interval_var = tk.DoubleVar(value=50)
        interval_spin = ttk.Spinbox(perf_frame, from_=10, to=200, increment=10, 
                                  textvariable=self.interval_var, width=8,
                                  command=self.update_interval)
        interval_spin.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(perf_frame, text="跳帧数:").pack(side=tk.LEFT)
        self.skip_var = tk.IntVar(value=2)
        skip_spin = ttk.Spinbox(perf_frame, from_=0, to=10, increment=1,
                              textvariable=self.skip_var, width=5)
        skip_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # 状态栏
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var).pack(side=tk.RIGHT)
        
        # 创建主要内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：传感器数值显示面板
        self.create_sensor_values_panel(content_frame)
        
        # 右侧：可视化面板
        self.create_visualization_panel(content_frame)
        
        # 初始化端口列表
        self.refresh_ports()
    
    def create_sensor_values_panel(self, parent):
        """创建传感器数值显示面板"""
        # 左侧传感器数值面板
        sensor_frame = ttk.LabelFrame(parent, text="传感器数值监控", padding=5)
        sensor_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # 传感器网格配置
        self.sensor_grid_size = (8, 8)  # 8x8传感器网格
        self.sensor_values = np.zeros(self.sensor_grid_size)
        
        # 创建传感器数值显示网格
        values_frame = ttk.Frame(sensor_frame)
        values_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 传感器数值标签网格
        self.sensor_labels = {}
        for i in range(self.sensor_grid_size[0]):
            for j in range(self.sensor_grid_size[1]):
                label = tk.Label(values_frame, text="0.00", 
                               width=6, height=2, relief=tk.RAISED,
                               bg='lightgray', font=('Courier', 8))
                label.grid(row=i, column=j, padx=1, pady=1)
                self.sensor_labels[(i, j)] = label
                
                # 添加鼠标悬停事件
                label.bind("<Enter>", lambda e, pos=(i,j): self.on_sensor_hover(e, pos))
                label.bind("<Leave>", self.on_sensor_leave)
        
        # 数值范围显示
        range_frame = ttk.Frame(sensor_frame)
        range_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(range_frame, text="数值范围:").pack(anchor=tk.W)
        self.min_value_var = tk.StringVar(value="最小: 0.00")
        self.max_value_var = tk.StringVar(value="最大: 0.00")
        self.avg_value_var = tk.StringVar(value="平均: 0.00")
        
        ttk.Label(range_frame, textvariable=self.min_value_var).pack(anchor=tk.W)
        ttk.Label(range_frame, textvariable=self.max_value_var).pack(anchor=tk.W)
        ttk.Label(range_frame, textvariable=self.avg_value_var).pack(anchor=tk.W)
        
        # 颜色映射说明
        color_frame = ttk.LabelFrame(sensor_frame, text="颜色映射", padding=5)
        color_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 创建颜色条
        color_canvas = tk.Canvas(color_frame, height=30, bg='white')
        color_canvas.pack(fill=tk.X, pady=5)
        
        # 绘制颜色条
        self.draw_color_bar(color_canvas)
        
        # 颜色说明
        ttk.Label(color_frame, text="蓝色: 低压力  绿色: 中等  红色: 高压力").pack()
        
        # 工具提示标签
        self.tooltip_var = tk.StringVar(value="悬停在传感器上查看详细信息")
        tooltip_label = ttk.Label(sensor_frame, textvariable=self.tooltip_var, 
                                foreground='blue', font=('Arial', 9))
        tooltip_label.pack(pady=(10, 0))

    def create_visualization_panel(self, parent):
        """创建可视化面板 - 优化版本"""
        viz_frame = ttk.LabelFrame(parent, text="实时可视化 (可缩放)", padding=5)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 添加缩放控制工具栏
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(toolbar_frame, text="缩放:").pack(side=tk.LEFT)
        
        # 缩放按钮
        ttk.Button(toolbar_frame, text="放大", command=self.zoom_in).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Button(toolbar_frame, text="缩小", command=self.zoom_out).pack(side=tk.LEFT, padx=(2, 2))
        ttk.Button(toolbar_frame, text="重置", command=self.reset_zoom).pack(side=tk.LEFT, padx=(2, 5))
        
        # 缩放比例显示
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(toolbar_frame, textvariable=self.zoom_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # 视图选择
        ttk.Label(toolbar_frame, text="视图:").pack(side=tk.LEFT, padx=(20, 5))
        self.view_mode_var = tk.StringVar(value="全部")
        view_combo = ttk.Combobox(toolbar_frame, textvariable=self.view_mode_var,
                                values=["全部", "原始数据", "检测结果", "3D视图", "热力图"], 
                                width=10, state="readonly")
        view_combo.pack(side=tk.LEFT)
        view_combo.bind('<<ComboboxSelected>>', self.on_view_mode_change)
        
        # 创建matplotlib图形 - 优化配置
        self.fig = Figure(figsize=(12, 8), dpi=80, facecolor='white')
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # 动态调整子图布局
        self.update_subplot_layout()
        
        # 优化颜色映射
        self.turbo_cmap = plt.cm.turbo
        
        # 创建画布 - 优化配置
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 添加鼠标事件处理
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        
        # 初始化数据存储
        self.trend_data = {
            'timestamps': deque(maxlen=100), 
            'nodule_counts': deque(maxlen=100),
            'high_prob': deque(maxlen=100),
            'medium_prob': deque(maxlen=100),
            'low_prob': deque(maxlen=100)
        }
        self.stats_data = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
    
    def draw_color_bar(self, canvas):
        """绘制颜色条"""
        canvas.delete("all")
        width = canvas.winfo_reqwidth()
        if width <= 1:
            width = 200
        height = 30
        
        # 绘制渐变色条
        for i in range(width):
            ratio = i / width
            # 从蓝色到绿色到红色的渐变
            if ratio < 0.5:
                r = int(255 * (ratio * 2))
                g = 255
                b = int(255 * (1 - ratio * 2))
            else:
                r = 255
                g = int(255 * (2 - ratio * 2))
                b = 0
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            canvas.create_line(i, 5, i, height-5, fill=color)
        
        # 添加刻度标签
        canvas.create_text(10, height//2, text="0", anchor="w")
        canvas.create_text(width//2, height//2, text="中", anchor="center")
        canvas.create_text(width-10, height//2, text="高", anchor="e")

    def on_sensor_hover(self, event, pos):
        """传感器悬停事件"""
        i, j = pos
        value = self.sensor_values[i, j]
        self.tooltip_var.set(f"传感器[{i},{j}]: {value:.3f} Pa")
        
        # 高亮显示当前传感器
        event.widget.config(relief=tk.SUNKEN, bg='lightyellow')

    def on_sensor_leave(self, event):
        """传感器离开事件"""
        self.tooltip_var.set("悬停在传感器上查看详细信息")
        event.widget.config(relief=tk.RAISED, bg=self.get_sensor_color(0))

    def get_sensor_color(self, value):
        """根据传感器数值获取颜色"""
        if value == 0:
            return 'lightgray'
        
        # 归一化值到0-1范围
        normalized = min(max(value / 100.0, 0), 1)  # 假设最大值为100
        
        if normalized < 0.3:
            return '#E3F2FD'  # 浅蓝色
        elif normalized < 0.6:
            return '#FFF3E0'  # 浅橙色
        else:
            return '#FFEBEE'  # 浅红色

    def update_sensor_values(self, matrix):
        """更新传感器数值显示"""
        try:
            # 调整矩阵大小以匹配传感器网格
            if matrix.shape != self.sensor_grid_size:
                from scipy import ndimage
                matrix = ndimage.zoom(matrix, 
                                    (self.sensor_grid_size[0]/matrix.shape[0], 
                                     self.sensor_grid_size[1]/matrix.shape[1]))
            
            self.sensor_values = matrix
            
            # 更新每个传感器标签
            for i in range(self.sensor_grid_size[0]):
                for j in range(self.sensor_grid_size[1]):
                    value = matrix[i, j]
                    label = self.sensor_labels[(i, j)]
                    
                    # 更新文本
                    label.config(text=f"{value:.2f}")
                    
                    # 更新颜色
                    color = self.get_sensor_color(value)
                    label.config(bg=color)
            
            # 更新统计信息
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            avg_val = np.mean(matrix)
            
            self.min_value_var.set(f"最小: {min_val:.3f}")
            self.max_value_var.set(f"最大: {max_val:.3f}")
            self.avg_value_var.set(f"平均: {avg_val:.3f}")
            
        except Exception as e:
            print(f"传感器数值更新错误: {e}")

    def update_subplot_layout(self):
        """根据视图模式更新子图布局"""
        self.fig.clear()
        
        view_mode = self.view_mode_var.get()
        
        if view_mode == "全部":
            self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                                   wspace=0.3, hspace=0.4)
            # 1. 原始数据
            self.ax_original = self.fig.add_subplot(2, 3, 1)
            # 2. 插值后特征提取图像
            self.ax_feature = self.fig.add_subplot(2, 3, 2)
            # 3. AI关键帧检测图像 (标记可疑区域)
            self.ax_detection = self.fig.add_subplot(2, 3, 3)
            # 4. 当前区域出现结节的概率
            self.ax_prob_gauge = self.fig.add_subplot(2, 3, 4)
            # 5. 大小概率分布
            self.ax_size_dist = self.fig.add_subplot(2, 3, 5)
            # 6. 深度概率分布
            self.ax_depth_dist = self.fig.add_subplot(2, 3, 6)
            
            self.ax_3d = None
            self.ax_trend = None
            self.ax_stats = None
        elif view_mode == "原始数据":
            self.ax_original = self.fig.add_subplot(1, 1, 1)
            self.ax_detection = None
            self.ax_contour = None
            self.ax_3d = None
            self.ax_trend = None
            self.ax_stats = None
        elif view_mode == "检测结果":
            self.ax_detection = self.fig.add_subplot(1, 1, 1)
            self.ax_original = None
            self.ax_contour = None
            self.ax_3d = None
            self.ax_trend = None
            self.ax_stats = None
        elif view_mode == "3D视图":
            self.ax_3d = self.fig.add_subplot(1, 1, 1, projection='3d')
            self.ax_original = None
            self.ax_detection = None
            self.ax_contour = None
            self.ax_trend = None
            self.ax_stats = None
        elif view_mode == "热力图":
            self.ax_heatmap = self.fig.add_subplot(1, 1, 1)
            self.ax_original = None
            self.ax_detection = None
            self.ax_contour = None
            self.ax_3d = None
            self.ax_trend = None
            self.ax_stats = None

    def on_view_mode_change(self, event=None):
        """视图模式改变事件"""
        self.update_subplot_layout()
        # 重新显示当前数据
        if hasattr(self, 'data') and self.data is not None:
            self.update_display()
        else:
            self.canvas.draw()

    def zoom_in(self):
        """放大图像"""
        self.zoom_level *= 1.2
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
        self.apply_zoom()

    def zoom_out(self):
        """缩小图像"""
        self.zoom_level /= 1.2
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
        self.apply_zoom()

    def reset_zoom(self):
        """重置缩放"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.zoom_var.set("100%")
        self.apply_zoom()

    def apply_zoom(self):
        """应用缩放设置"""
        try:
            # 获取当前所有轴
            axes = [ax for ax in [self.ax_original, self.ax_detection, self.ax_contour] if ax is not None]
            
            for ax in axes:
                # 获取原始数据范围
                if hasattr(self, 'original_xlim') and hasattr(self, 'original_ylim'):
                    xlim = self.original_xlim
                    ylim = self.original_ylim
                else:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    # 保存原始范围
                    self.original_xlim = xlim
                    self.original_ylim = ylim
                
                # 计算缩放后的范围
                x_center = (xlim[0] + xlim[1]) / 2 + self.pan_offset[0]
                y_center = (ylim[0] + ylim[1]) / 2 + self.pan_offset[1]
                
                x_range = (xlim[1] - xlim[0]) / self.zoom_level
                y_range = (ylim[1] - ylim[0]) / self.zoom_level
                
                # 设置新的显示范围
                ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
                ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            
            # 更新zoom_factor用于网格显示
            self.zoom_factor = self.zoom_level
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"缩放应用错误: {e}")

    def on_canvas_click(self, event):
        """画布点击事件"""
        if event.inaxes:
            self.last_click_pos = (event.xdata, event.ydata)

    def on_canvas_scroll(self, event):
        """画布滚轮事件"""
        if event.inaxes:
            # 滚轮缩放
            if event.button == 'up':
                self.zoom_level *= 1.1
            elif event.button == 'down':
                self.zoom_level /= 1.1
            
            self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
            self.apply_zoom()

    def on_canvas_motion(self, event):
        """画布鼠标移动事件"""
        try:
            if event.inaxes:
                x, y = int(event.xdata), int(event.ydata)
                
                # 检查是否有传感器数据
                if hasattr(self, 'sensor_values') and self.sensor_values is not None:
                    rows, cols = self.sensor_values.shape
                    if 0 <= x < cols and 0 <= y < rows:
                        value = self.sensor_values[y, x]
                        # 更新工具提示
                        tooltip_text = f"传感器[{y},{x}]: {value:.3f} Pa"
                        self.tooltip_var.set(tooltip_text)
                        
                        # 高亮显示当前传感器位置
                        self.highlight_sensor_position(x, y)
                    else:
                        self.tooltip_var.set("鼠标位置信息")
                else:
                    self.tooltip_var.set("等待传感器数据...")
        except Exception as e:
            self.tooltip_var.set("鼠标位置信息")
    
    def highlight_sensor_position(self, x, y):
        """高亮显示传感器位置"""
        try:
            # 清除之前的高亮
            if hasattr(self, 'highlight_artists'):
                for artist in self.highlight_artists:
                    try:
                        artist.remove()
                    except:
                        pass
            
            self.highlight_artists = []
            
            # 在所有相关轴上添加高亮
            axes = [self.ax_original, self.ax_detection, self.ax_contour]
            for ax in axes:
                if ax is not None:
                    # 添加高亮圆圈
                    circle = plt.Circle((x, y), 0.3, fill=False, 
                                      color='yellow', linewidth=2, alpha=0.8)
                    ax.add_patch(circle)
                    self.highlight_artists.append(circle)
            
            # 刷新画布
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"高亮显示错误: {e}")

    def update_interval(self):
        """更新处理间隔"""
        self.plot_interval = self.interval_var.get() / 1000.0
    
    def start_performance_monitor(self):
        """启动性能监控"""
        def monitor():
            while self.is_realtime_processing and hasattr(self, 'root') and self.root.winfo_exists():
                try:
                    if len(self.fps_counter) > 1:
                        avg_fps = len(self.fps_counter) / (time.time() - self.fps_counter[0])
                        # 安全的GUI更新
                        if hasattr(self, 'root') and self.root.winfo_exists():
                            self.root.after(0, lambda fps=avg_fps: self.safe_update_fps(fps))
                    time.sleep(1)
                except Exception as e:
                    print(f"性能监控错误: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def toggle_threshold(self):
        """切换阈值范围启用状态"""
        self.enable_threshold = self.threshold_enable_var.get()
        if self.enable_threshold:
            self.stress_min_threshold = self.min_threshold_var.get()
            self.stress_max_threshold = self.max_threshold_var.get()
            self.threshold_status_var.set(f"已启用 ({self.stress_min_threshold:.1f}-{self.stress_max_threshold:.1f})")
            self.threshold_status_label.config(foreground="green")
        else:
            self.threshold_status_var.set("未启用")
            self.threshold_status_label.config(foreground="gray")
        
        print(f"应力阈值范围{'启用' if self.enable_threshold else '禁用'}: {self.stress_min_threshold:.1f}-{self.stress_max_threshold:.1f}")
    
    def update_min_threshold(self):
        """更新最小阈值设置"""
        self.stress_min_threshold = self.min_threshold_var.get()
        if self.enable_threshold:
            self.threshold_status_var.set(f"已启用 ({self.stress_min_threshold:.1f}-{self.stress_max_threshold:.1f})")
        print(f"最小应力阈值更新为: {self.stress_min_threshold:.1f}")
    
    def update_max_threshold(self):
        """更新最大阈值设置"""
        self.stress_max_threshold = self.max_threshold_var.get()
        if self.enable_threshold:
            self.threshold_status_var.set(f"已启用 ({self.stress_min_threshold:.1f}-{self.stress_max_threshold:.1f})")
        print(f"最大应力阈值更新为: {self.stress_max_threshold:.1f}")
    
    def update_nodule_sphericity(self):
        """更新结节原形度参数"""
        self.nodule_sphericity = self.sphericity_var.get()
        if hasattr(self, 'elastography_system'):
            self.elastography_system.update_nodule_parameters(
                sphericity=self.nodule_sphericity,
                size=self.nodule_size
            )
        print(f"结节原形度更新为: {self.nodule_sphericity:.2f}")
    
    def update_nodule_size(self):
        """更新结节尺寸参数"""
        self.nodule_size = self.size_var.get()
        if hasattr(self, 'elastography_system'):
            self.elastography_system.update_nodule_parameters(
                sphericity=self.nodule_sphericity,
                size=self.nodule_size
            )
        print(f"结节尺寸更新为: {self.nodule_size:.1f} mm")
    
    def toggle_elastography(self):
        """切换弹性成像分析模式"""
        self.enable_elastography = self.elastography_var.get()
        status = "启用" if self.enable_elastography else "禁用"
        print(f"弹性成像分析已{status}")
        
        # 如果禁用弹性成像，清除相关显示
        if not self.enable_elastography and hasattr(self, 'ax_2d'):
            # 清除边界标记
            for artist in getattr(self.ax_2d, '_elastography_artists', []):
                try:
                    artist.remove()
                except:
                    pass
            self.ax_2d._elastography_artists = []
            self.canvas.draw_idle()
    
    def apply_stress_threshold(self, matrix):
        """应用应力阈值范围处理"""
        try:
            if self.enable_threshold:
                # 将小于最小阈值的值设为0，大于最大阈值的值设为最大阈值
                processed_matrix = np.where(matrix < self.stress_min_threshold, 0, matrix)
                processed_matrix = np.where(processed_matrix > self.stress_max_threshold, self.stress_max_threshold, processed_matrix)
                return processed_matrix
            return matrix
        except Exception as e:
            print(f"应力阈值范围处理错误: {e}")
            return matrix

    def on_closing(self):
        """窗口关闭处理"""
        try:
            # 停止实时处理
            self.is_realtime_processing = False
            
            # 关闭串口连接
            if hasattr(self, 'serial_port') and self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                print("串口连接已关闭")
            
            # 清理队列
            if hasattr(self, 'data_queue'):
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
            
            # 等待线程结束
            time.sleep(0.1)
            
            print("应用程序正在关闭...")
            
        except Exception as e:
            print(f"关闭处理错误: {e}")
        finally:
            # 销毁窗口
            self.root.quit()
            self.root.destroy()

    def safe_update_fps(self, fps):
        """安全更新FPS显示"""
        try:
            if hasattr(self, 'fps_var') and hasattr(self, 'root') and self.root.winfo_exists():
                self.fps_var.set(f"FPS: {fps:.1f}")
        except Exception as e:
            print(f"FPS更新错误: {e}")
    
    def refresh_ports(self):
        """刷新串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
    
    def toggle_serial(self):
        """切换串口连接状态"""
        if not self.is_serial_connected:
            self.connect_serial()
        else:
            self.disconnect_serial()
    
    def connect_serial(self):
        """连接串口 - 优化版本"""
        port = self.port_var.get()
        if not port:
            messagebox.showwarning("警告", "请选择串口")
            return
        
        try:
            # 优化串口配置
            self.serial_port = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=0.001,  # 非阻塞读取
                write_timeout=0.001
            )
            
            self.is_serial_connected = True
            self.serial_btn.config(text="断开串口")
            self.status_var.set(f"已连接到 {port}")
            
            # 启动优化的串口读取线程
            self.serial_thread = threading.Thread(target=self.optimized_serial_loop, daemon=True)
            self.serial_thread.start()
            
            # 启动实时处理
            self.start_realtime_processing()
            
        except Exception as e:
            messagebox.showerror("连接失败", f"无法连接到串口: {str(e)}")
    
    def disconnect_serial(self):
        """断开串口连接"""
        self.is_serial_connected = False
        self.is_realtime_processing = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.serial_btn.config(text="连接串口")
        self.status_var.set("已断开连接")
    
    def optimized_serial_loop(self):
        """优化的串口读取循环"""
        buffer = bytearray()
        
        while self.is_serial_connected and self.serial_port and self.serial_port.is_open:
            try:
                # 批量读取数据
                avail = self.serial_port.in_waiting
                if avail > 0:
                    chunk_size = min(avail, 4096)  # 增大读取块大小
                    data = self.serial_port.read(chunk_size)
                    buffer.extend(data)
                    
                    # 当缓冲区足够大时处理数据
                    if len(buffer) >= 1024:
                        # 添加到解析器
                        self.parser.add_data(bytes(buffer))
                        
                        # 清空缓冲区
                        buffer.clear()
                        
                        # 非阻塞队列操作
                        try:
                            self.data_queue.put_nowait(True)  # 只发送信号
                        except queue.Full:
                            # 清空队列保持低延迟
                            while not self.data_queue.empty():
                                try:
                                    self.data_queue.get_nowait()
                                except queue.Empty:
                                    break
                            self.data_queue.put_nowait(True)
                
                time.sleep(0.0001)  # 极短睡眠
                
            except Exception as e:
                print(f"串口读取错误: {e}")
                break
    
    def start_realtime_processing(self):
        """启动实时数据处理循环 - 优化版本"""
        self.is_realtime_processing = True
        self.process_realtime_data()
    
    def process_realtime_data(self):
        """处理实时数据 - 高性能版本"""
        if not self.is_realtime_processing or not self.is_serial_connected:
            return
        
        start_time = time.time()
        
        try:
            # 跳帧处理
            self.frame_counter += 1
            if self.frame_counter % (self.skip_var.get() + 1) != 0:
                self.schedule_next_processing()
                return
            
            # 检查是否有新数据
            if self.data_queue.empty():
                self.schedule_next_processing()
                return
            
            # 清空队列，只处理最新数据
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 获取最新解析的帧数据
            latest_frame = self.parser.get_latest()
            
            if latest_frame is not None:
                matrix = latest_frame['matrix']
                timestamp = latest_frame['timestamp']
                
                # 应用应力阈值处理
                matrix = self.apply_stress_threshold(matrix)
                
                # AI深度学习检测 (实时处理每一帧)
                if self.use_dl_detection and self.dl_var.get() and self.dl_model:
                    try:
                        # 1. Store raw frame in buffer
                        self.frame_buffer.append(matrix)
                        
                        if len(self.frame_buffer) == self.seq_len:
                            # 2. Prepare Input Tensor
                            seq_raw = np.array(self.frame_buffer) # (Seq, 12, 8)
                            
                            # Intensity Stats
                            avg_intensity = np.mean(seq_raw)
                            max_intensity = np.max(seq_raw)
                            std_intensity = np.std(seq_raw)
                            intensity_stats = torch.tensor([[avg_intensity, max_intensity, std_intensity]], dtype=torch.float32).to(self.device)
                            
                            # Normalize Sequence
                            seq_norm = np.zeros((self.seq_len, 1, 12, 8), dtype=np.float32)
                            for i in range(self.seq_len):
                                fr = seq_raw[i]
                                mn, mx = fr.min(), fr.max()
                                if mx - mn > 1e-6:
                                    fr = (fr - mn) / (mx - mn)
                                else:
                                    fr = fr - mn
                                seq_norm[i, 0] = fr
                            
                            # To Tensor: (1, Seq, 1, 12, 8)
                            input_tensor = torch.tensor(seq_norm).unsqueeze(0).to(self.device)
                            
                            # 3. Inference
                            with torch.no_grad():
                                prob, size, depth = self.dl_model(input_tensor, intensity_stats)
                                self.dl_probability = prob.item()
                                self.dl_pred_size = size.item()
                                self.dl_pred_depth = depth.item()
                                
                        # 记录概率用于曲线显示
                        if not hasattr(self, 'prob_history'):
                            self.prob_history = []
                        self.prob_history.append(self.dl_probability)
                        if len(self.prob_history) > 100: # 限制长度
                            self.prob_history.pop(0)
                            
                        # 更新概率曲线
                        self.update_prob_curve()
                        
                    except Exception as e:
                        print(f"AI推理错误: {e}")
                        self.dl_probability = 0.0
                else:
                    self.dl_probability = 0.0
                
                # 计算矩阵哈希用于缓存
                matrix_hash = hash(matrix.tobytes())
                
                # 检查缓存
                if matrix_hash == self.last_matrix_hash:
                    self.schedule_next_processing()
                    return
                
                self.last_matrix_hash = matrix_hash
                
                # 执行结节检测（可能使用缓存）
                if matrix_hash in self.visualization_cache:
                    cached_result = self.visualization_cache[matrix_hash]
                    normalized, nodule_mask, nodules = cached_result
                    prob_map = normalized  # 使用normalized作为prob_map
                else:
                    if self.use_enhanced_detection and self.enhanced_detector.is_trained:
                        # 使用增强检测系统
                        enhanced_result = self.enhanced_detector.process_frame(matrix.flatten(), timestamp)
                        if enhanced_result:
                            # 转换增强检测结果为标准格式
                            normalized = enhanced_result['matrix_data']['normalized_matrix']
                            prob_map = np.ones_like(normalized) * enhanced_result['combined_probability']
                            
                            # 创建简化的结节列表
                            nodules = []
                            if enhanced_result['combined_probability'] > 0.5:
                                center = np.unravel_index(np.argmax(normalized), normalized.shape)
                                nodules.append({
                                    'centroid': center,
                                    'risk_score': enhanced_result['combined_probability'],
                                    'area': 1.0,
                                    'enhanced': True
                                })
                            
                            nodule_mask = prob_map > 0.5
                        else:
                            # 回退到标准检测
                            normalized, nodule_mask, nodules = self.detector.advanced_nodule_detection(
                                matrix, timestamp
                            )
                            prob_map = normalized
                    else:
                        # 使用标准检测
                        normalized, nodule_mask, nodules = self.detector.advanced_nodule_detection(
                            matrix, timestamp
                        )
                        prob_map = normalized  # 使用normalized作为prob_map
                    
                    # 如果启用弹性成像分析，进行额外处理
                    if self.enable_elastography and hasattr(self, 'elastography_system'):
                        try:
                            # 执行弹性成像分析
                            elastography_result = self.elastography_system.analyze_tissue_elasticity(
                                matrix, self.nodule_sphericity, self.nodule_size
                            )
                            
                            if elastography_result:
                                # 更新结节信息，添加弹性成像数据
                                nodule_probabilities = elastography_result.get('nodule_probabilities', [])
                                for i, nodule in enumerate(nodules):
                                    if i < len(nodule_probabilities):
                                        prob_data = nodule_probabilities[i]
                                        nodule['nodule_probability'] = prob_data['nodule_probability']
                                        nodule['elasticity_contrast'] = prob_data.get('elasticity_contrast', 0)
                                        nodule['boundary_coords'] = prob_data.get('boundary_coords', [])
                                        nodule['youngs_modulus'] = prob_data.get('youngs_modulus', 0)
                                        # 用结节概率替换风险评分
                                        nodule['risk_score'] = prob_data['nodule_probability']
                                
                                # 使用弹性对比度作为概率图
                                prob_map = elastography_result.get('elasticity_contrast', prob_map)
                                
                        except Exception as e:
                            print(f"弹性成像分析错误: {e}")
                    
                    # 缓存结果
                    if len(self.visualization_cache) < self.cache_size_limit:
                        self.visualization_cache[matrix_hash] = (normalized, nodule_mask, nodules)
                
                # 异步更新可视化
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after_idle(self.update_realtime_plots_optimized, 
                                       normalized, nodule_mask, nodules, prob_map)
                
                # 更新传感器数值显示
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.after_idle(self.update_sensor_values, matrix)
                
                # 更新统计信息
                self.update_statistics_optimized(nodules, timestamp)
                
                # 更新状态
                self.status_var.set(f"实时处理中 - 检出结节: {len(nodules)}个")
                
                # 更新FPS计数
                self.fps_counter.append(time.time())
                
        except Exception as e:
            print(f"实时数据处理错误: {e}")
            self.status_var.set(f"实时处理错误: {str(e)}")
        
        # 调度下次处理
        self.schedule_next_processing()
    
    def schedule_next_processing(self):
        """调度下次处理"""
        try:
            if self.is_realtime_processing and hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(int(self.plot_interval * 1000), self.process_realtime_data)
        except Exception as e:
            print(f"调度处理错误: {e}")
            self.is_realtime_processing = False
    
    def update_realtime_plots_optimized(self, normalized, nodule_mask, nodules, prob_map):
        """优化的实时可视化更新"""
        try:
            # 获取当前视图模式
            view_mode = self.view_mode_var.get()
            
            # 根据视图模式决定更新哪些轴
            if view_mode == "全部":
                # 清除所有轴但保留配置
                if self.ax_original: self.ax_original.clear()
                if self.ax_feature: self.ax_feature.clear()
                if self.ax_detection: self.ax_detection.clear()
                if self.ax_prob_gauge: self.ax_prob_gauge.clear()
                if self.ax_size_dist: self.ax_size_dist.clear()
                if self.ax_depth_dist: self.ax_depth_dist.clear()
                
                # 1. 原始数据
                self._update_raw_view(normalized)
                
                # 2. 插值后特征提取图像
                self._update_feature_view(normalized, nodules)
                
                # 3. AI关键帧检测图像
                self._update_ai_detection_view(prob_map, normalized, nodules)
                
                # 4. 概率
                self._update_prob_view()
                
                # 5 & 6. 分布
                self._update_dist_views()
                
                # Update Probability Curve (Background history, optional)
                if hasattr(self, 'update_prob_curve'):
                    self.update_prob_curve()
                
            elif view_mode == "原始数据":
                if self.ax_original:
                    self.ax_original.clear()
                    self._update_original_view(normalized)
                    
            elif view_mode == "检测结果":
                if self.ax_detection:
                    self.ax_detection.clear()
                    self._update_detection_view(prob_map, nodules)
                    
            elif view_mode == "3D视图":
                if self.ax_3d:
                    self.update_3d_plot_optimized(normalized)
                    
            elif view_mode == "热力图":
                if hasattr(self, 'ax_heatmap') and self.ax_heatmap:
                    self.ax_heatmap.clear()
                    self._update_heatmap_view(normalized, prob_map, nodules)
            
            # 快速刷新画布
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"可视化更新错误: {e}")
    
    def _interpolate_matrix(self, matrix, factor=5):
        """对矩阵进行插值处理"""
        from scipy import ndimage
        # 归一化
        mn, mx = matrix.min(), matrix.max()
        if mx - mn > 1e-6:
            norm = (matrix - mn) / (mx - mn)
        else:
            norm = matrix - mn
        
        # 插值 12x8 -> 60x40 (factor=5)
        return ndimage.zoom(norm, factor, order=3) # Bicubic

    def _update_raw_view(self, normalized):
        """1. 原始数据 (Raw)"""
        if self.ax_original:
            self.ax_original.clear()
            # Show raw pixelated
            self.ax_original.imshow(normalized, cmap=self.turbo_cmap, aspect='auto', interpolation='nearest')
            self.ax_original.set_title('1. 原始数据 (12x8)', fontsize=9)
            # Add grid
            self.ax_original.set_xticks(np.arange(-.5, 8, 1), minor=True)
            self.ax_original.set_yticks(np.arange(-.5, 12, 1), minor=True)
            self.ax_original.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    def _update_feature_view(self, normalized, nodules):
        """2. 插值后特征提取图像"""
        if self.ax_feature:
            self.ax_feature.clear()
            img = self._interpolate_matrix(normalized)
            self.ax_feature.imshow(img, cmap=self.turbo_cmap, aspect='auto', interpolation='bicubic')
            self.ax_feature.set_title('2. 特征提取 (插值)', fontsize=9)
            
            # Draw feature contours
            try:
                self.ax_feature.contour(img, levels=[0.4, 0.6, 0.8], colors='white', alpha=0.5, linewidths=0.5)
            except:
                pass

    def _update_ai_detection_view(self, prob_map, normalized, nodules):
        """3. AI关键帧检测图像 (标记可疑区域)"""
        if self.ax_detection:
            self.ax_detection.clear()
            img = self._interpolate_matrix(normalized)
            self.ax_detection.imshow(img, cmap=self.turbo_cmap, aspect='auto', interpolation='bicubic', alpha=0.8)
            
            title = '3. AI检测标记'
            prob = getattr(self, 'dl_probability', 0.0)
            if prob > 0.5:
                # Draw red box
                rect = plt.Rectangle((0,0), img.shape[1], img.shape[0], linewidth=3, edgecolor='r', facecolor='none')
                self.ax_detection.add_patch(rect)
                title += f" [DETECTED P={prob:.2f}]"
            else:
                title += f" [P={prob:.2f}]"
            
            self.ax_detection.set_title(title, fontsize=9, color='red' if prob > 0.5 else 'black')
            
            # Mark nodules
            scale_y = img.shape[0] / 12
            scale_x = img.shape[1] / 8
            for n in nodules:
                if 'centroid' in n:
                    cy, cx = n['centroid']
                    circle = plt.Circle((cx*scale_x, cy*scale_y), radius=5, color='yellow', fill=False, linewidth=2)
                    self.ax_detection.add_patch(circle)

    def _update_prob_view(self):
        """4. 当前区域出现结节的概率"""
        if self.ax_prob_gauge:
            prob = getattr(self, 'dl_probability', 0.0)
            self.ax_prob_gauge.clear()
            
            # Draw a gauge-like bar
            self.ax_prob_gauge.barh([0], [prob], color='red' if prob > 0.5 else 'green', height=0.5)
            self.ax_prob_gauge.set_xlim(0, 1)
            self.ax_prob_gauge.set_ylim(-0.5, 0.5)
            self.ax_prob_gauge.set_yticks([])
            self.ax_prob_gauge.set_xticks([0, 0.5, 1.0])
            self.ax_prob_gauge.set_title(f'4. 结节概率: {prob:.1%}', fontsize=10, fontweight='bold')
            
            # Add threshold line
            self.ax_prob_gauge.axvline(0.5, color='black', linestyle='--')

    def _update_dist_views(self):
        """5 & 6. 大小和深度概率分布"""
        # 5. Size
        if self.ax_size_dist:
            self.ax_size_dist.clear()
            if hasattr(self, 'size_dist_data') and self.size_dist_data:
                sizes, probs = self.size_dist_data
                self.ax_size_dist.plot(sizes, probs, 'b-', linewidth=2)
                self.ax_size_dist.fill_between(sizes, probs, alpha=0.3, color='blue')
                
                # Mark prior
                prior = self.size_var.get()
                # Find prob at prior
                idx = (np.abs(sizes - prior)).argmin()
                prior_prob = probs[idx]
                
                self.ax_size_dist.axvline(prior, color='r', linestyle='--', label=f'输入: {prior}mm')
                self.ax_size_dist.plot(prior, prior_prob, 'ro')
                self.ax_size_dist.text(prior, prior_prob, f'{prior_prob:.2f}', color='red', fontsize=8)
                
                self.ax_size_dist.set_title(f'5. 大小准确率: {prior_prob:.1%}', fontsize=9)
                self.ax_size_dist.legend(fontsize=6, loc='upper right')
                self.ax_size_dist.set_ylim(0, 1.05)
                self.ax_size_dist.grid(True, alpha=0.3)
            else:
                self.ax_size_dist.text(0.5, 0.5, "分析中...", ha='center')
                self.ax_size_dist.set_title('5. 大小概率分布', fontsize=9)

        # 6. Depth
        if self.ax_depth_dist:
            self.ax_depth_dist.clear()
            if hasattr(self, 'depth_dist_data') and self.depth_dist_data:
                depths, probs = self.depth_dist_data
                self.ax_depth_dist.plot(depths, probs, 'g-', linewidth=2)
                self.ax_depth_dist.fill_between(depths, probs, alpha=0.3, color='green')
                
                prior = self.depth_var.get()
                idx = (np.abs(depths - prior)).argmin()
                prior_prob = probs[idx]
                
                self.ax_depth_dist.axvline(prior, color='r', linestyle='--', label=f'输入: {prior}mm')
                self.ax_depth_dist.plot(prior, prior_prob, 'ro')
                self.ax_depth_dist.text(prior, prior_prob, f'{prior_prob:.2f}', color='red', fontsize=8)
                
                self.ax_depth_dist.set_title(f'6. 深度准确率: {prior_prob:.1%}', fontsize=9)
                self.ax_depth_dist.legend(fontsize=6, loc='upper right')
                self.ax_depth_dist.set_ylim(0, 1.05)
                self.ax_depth_dist.grid(True, alpha=0.3)
            else:
                self.ax_depth_dist.text(0.5, 0.5, "分析中...", ha='center')
                self.ax_depth_dist.set_title('6. 深度概率分布', fontsize=9)

    def _update_original_view(self, normalized):
        """更新原始数据视图"""
        if self.ax_original:
            # 插值显示
            img = self._interpolate_matrix(normalized)
            im1 = self.ax_original.imshow(img, cmap=self.turbo_cmap, aspect='auto', 
                                        interpolation='bilinear') # Use bilinear for smooth display
            self.ax_original.set_title('原始数据 (插值)', fontsize=9)
            # self.add_sensor_grid_overlay(self.ax_original, normalized) # Grid overlay might mismatch interpolated image
    
    def _update_detection_view(self, prob_map, nodules):
        """更新检测结果视图"""
        if self.ax_detection:
            # 插值显示概率图
            img = self._interpolate_matrix(prob_map)
            im2 = self.ax_detection.imshow(img, cmap=self.turbo_cmap, aspect='auto',
                                         interpolation='bilinear')
            
            title = '结节检测'
            if self.use_dl_detection and hasattr(self, 'dl_probability'):
                dl_status = "异常" if self.dl_probability > 0.5 else "正常"
                # 根据概率改变标题颜色 (通过改变字符串前缀实现不了，需要操作ax title对象)
                title_color = 'red' if self.dl_probability > 0.5 else 'black'
                title += f" [AI: {dl_status} P={self.dl_probability:.2f}]"
                self.ax_detection.set_title(title, fontsize=10, color=title_color, fontweight='bold')
            else:
                self.ax_detection.set_title(title, fontsize=9)
            
            # self.add_sensor_grid_overlay(self.ax_detection, prob_map)
            
            # 标记结节 (需要坐标变换)
            # nodules coordinates are in 12x8 space, need to map to 60x40
            scale_y = 60 / 12
            scale_x = 40 / 8
            scaled_nodules = []
            for n in nodules:
                # Assuming n has 'centroid' (y, x) or 'bbox'
                if 'centroid' in n:
                    cy, cx = n['centroid']
                    scaled_nodules.append({'centroid': (cy * scale_y, cx * scale_x), 'risk_score': n.get('risk_score', 0)})
            
            self._mark_nodules(self.ax_detection, scaled_nodules)
    
    def _update_contour_view(self, normalized):
        """更新热力图视图"""
        if self.ax_contour:
            img = self._interpolate_matrix(normalized)
            self.ax_contour.imshow(img, cmap=self.turbo_cmap, aspect='auto',
                                 interpolation='bilinear')
            self.ax_contour.set_title('热力图 (插值)', fontsize=9)
            # self.add_sensor_grid_overlay(self.ax_contour, normalized)
    
    def _update_heatmap_view(self, normalized, prob_map, nodules):
        """更新专用热力图视图 - 与全部视图中的热力图保持一致"""
        if self.ax_heatmap:
            # 插值显示
            img = self._interpolate_matrix(normalized)
            im = self.ax_heatmap.imshow(img, cmap=self.turbo_cmap, aspect='auto',
                                      interpolation='bilinear')
            self.ax_heatmap.set_title('热力图 (插值)', fontsize=10)
            
            # self.add_sensor_grid_overlay(self.ax_heatmap, normalized)
            
            # 标记结节 (需要坐标变换)
            scale_y = 60 / 12
            scale_x = 40 / 8
            scaled_nodules = []
            for n in nodules:
                if 'centroid' in n:
                    cy, cx = n['centroid']
                    scaled_nodules.append({'centroid': (cy * scale_y, cx * scale_x), 'risk_score': n.get('risk_score', 0)})
            
            self._mark_nodules(self.ax_heatmap, scaled_nodules)
            
            # 添加颜色条
            try:
                # 移除之前的颜色条（如果存在）
                if hasattr(self, '_heatmap_colorbar') and self._heatmap_colorbar:
                    self._heatmap_colorbar.remove()
                
                # 添加新的颜色条
                self._heatmap_colorbar = plt.colorbar(im, ax=self.ax_heatmap, 
                                                    shrink=0.8, aspect=20)
                self._heatmap_colorbar.set_label('强度值', rotation=270, labelpad=15)
            except Exception as e:
                print(f"颜色条添加错误: {e}")
    
    def _mark_nodules(self, ax, nodules):
        """在指定轴上标记结节"""
        # 清除之前的弹性成像标记
        if not hasattr(ax, '_elastography_artists'):
            ax._elastography_artists = []
        
        for artist in ax._elastography_artists:
            try:
                artist.remove()
            except:
                pass
        ax._elastography_artists = []
        
        for nodule in nodules[:10]:  # 限制显示数量
            x, y = nodule['centroid']  # 使用centroid而不是position
            risk = nodule['risk_score']
            
            # 根据是否有弹性成像数据选择显示方式
            if 'nodule_probability' in nodule and self.enable_elastography:
                # 显示结节概率而不是风险评分
                prob = nodule['nodule_probability']
                color = 'red' if prob > 0.8 else 'orange' if prob > 0.6 else 'yellow'
                marker = ax.plot(y, x, 'o', color=color, markersize=8, 
                              markeredgecolor='black', markeredgewidth=2)
                ax._elastography_artists.extend(marker)
                
                # 添加概率标签
                text = ax.text(y+0.5, x, f'{prob:.2f}', 
                            fontsize=8, color='white', weight='bold')
                ax._elastography_artists.append(text)
                
                # 如果有边界坐标，绘制边界
                if 'boundary_coords' in nodule and nodule['boundary_coords'] is not None:
                    boundary = nodule['boundary_coords']
                    if len(boundary) > 0:
                        boundary_x = [coord[0] for coord in boundary]
                        boundary_y = [coord[1] for coord in boundary]
                        # 闭合边界
                        boundary_x.append(boundary_x[0])
                        boundary_y.append(boundary_y[0])
                        
                        line = ax.plot(boundary_y, boundary_x, 
                                    color='cyan', linewidth=2, alpha=0.8)
                        ax._elastography_artists.extend(line)
            else:
                # 传统风险评分显示
                color = 'red' if risk > 0.8 else 'orange' if risk > 0.6 else 'yellow'
                marker = ax.plot(y, x, 'o', color=color, markersize=6, 
                              markeredgecolor='black', markeredgewidth=1)
                ax._elastography_artists.extend(marker)
    
    def update_3d_plot_optimized(self, normalized):
        """优化的3D图更新"""
        try:
            self.ax_3d.clear()
            
            # 降采样以提高性能
            step = max(1, normalized.shape[0] // 8)
            x = np.arange(0, normalized.shape[1], step)
            y = np.arange(0, normalized.shape[0], step)
            X, Y = np.meshgrid(x, y)
            Z = normalized[::step, ::step]
            
            # 简化的3D表面图
            self.ax_3d.plot_surface(X, Y, Z, cmap=self.turbo_cmap, alpha=0.7,
                                  linewidth=0, antialiased=False)
            self.ax_3d.set_title('3D视图', fontsize=9)
            
        except Exception as e:
            print(f"3D图更新错误: {e}")
    
    def scan_full_file(self):
        """全文件AI扫描，寻找最大概率帧"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件")
            return
            
        if not self.use_dl_detection or not self.dl_model:
            messagebox.showwarning("警告", "AI模型未加载")
            return
            
        try:
            # Show progress
            progress_win = tk.Toplevel(self.root)
            progress_win.title("正在扫描...")
            progress_win.geometry("300x100")
            progress_bar = ttk.Progressbar(progress_win, length=200, mode='determinate')
            progress_bar.pack(pady=20)
            status_lbl = ttk.Label(progress_win, text="正在分析全文件时序特征...")
            status_lbl.pack()
            
            self.root.update()
            
            total_frames = len(self.data)
            self.prob_history = [0.0] * total_frames
            
            max_prob = -1.0
            max_frame = 0
            
            # Simple loop
            for i in range(total_frames):
                # Prepare sequence (10 frames)
                if i >= 9:
                    seq_raw = self.data[i-9 : i+1]
                else:
                    pad_len = 10 - (i + 1)
                    first_frame = self.data[[0]]
                    pad = np.repeat(first_frame, pad_len, axis=0)
                    current_slice = self.data[0 : i+1]
                    seq_raw = np.vstack([pad, current_slice])
                
                # Preprocess
                seq_raw_reshaped = []
                for j in range(10):
                    fr = seq_raw[j]
                    if fr.size > 96: fr = fr[-96:]
                    seq_raw_reshaped.append(fr.reshape(12, 8))
                seq_raw_reshaped = np.array(seq_raw_reshaped)
                
                # Stats
                avg_intensity = np.mean(seq_raw_reshaped)
                max_intensity = np.max(seq_raw_reshaped)
                std_intensity = np.std(seq_raw_reshaped)
                intensity_stats = torch.tensor([[avg_intensity, max_intensity, std_intensity]], dtype=torch.float32).to(self.device)
                
                # Norm
                seq_norm = np.zeros((10, 1, 12, 8), dtype=np.float32)
                for j in range(10):
                    fr = seq_raw_reshaped[j]
                    mn, mx = fr.min(), fr.max()
                    if mx - mn > 1e-6:
                        fr = (fr - mn) / (mx - mn)
                    else:
                        fr = fr - mn
                    seq_norm[j, 0] = fr
                    
                input_tensor = torch.tensor(seq_norm).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    prob, _, _ = self.dl_model(input_tensor, intensity_stats)
                    prob_val = prob.item()
                
                self.prob_history[i] = prob_val
                
                if prob_val > max_prob:
                    max_prob = prob_val
                    max_frame = i
                
                if i % 10 == 0:
                    progress_bar['value'] = (i / total_frames) * 100
                    progress_win.update()
            
            progress_win.destroy()
            
            # Jump to max frame
            self.current_frame = max_frame
            self.frame_var.set(max_frame)
            self.update_display()
            
            messagebox.showinfo("扫描完成", f"扫描完成！\n发现最大结节概率: {max_prob:.2%}\n位于第 {max_frame} 帧")
            
        except Exception as e:
            print(f"扫描错误: {e}")
            messagebox.showerror("错误", f"扫描失败: {str(e)}")

    def update_prob_curve(self):
        """更新概率曲线 - 增强版"""
        if not hasattr(self, 'ax_prob') or self.ax_prob is None:
            return
            
        try:
            self.ax_prob.clear()
            
            # 基础设置（确保即使无数据也显示坐标轴）
            self.ax_prob.set_title('AI 结节概率实时曲线', fontsize=9)
            self.ax_prob.set_xlabel('帧序号 (Frame)')
            self.ax_prob.set_ylabel('结节概率 (Probability)')
            self.ax_prob.set_ylim(-0.05, 1.05)
            self.ax_prob.grid(True, alpha=0.3, linestyle='--')
            
            # 绘制阈值线
            self.ax_prob.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='阈值 (Threshold 0.5)')
            
            if hasattr(self, 'prob_history') and self.prob_history:
                x = np.arange(len(self.prob_history))
                y = np.array(self.prob_history)
                
                # 绘制主曲线
                self.ax_prob.plot(x, y, 'b-', linewidth=1.5, label='结节概率 (Nodule Prob)')
                
                # 填充曲线下方区域（增加视觉效果）
                self.ax_prob.fill_between(x, y, alpha=0.1, color='blue')
                
                # 标记当前帧
                if hasattr(self, 'current_frame'):
                    self.ax_prob.axvline(x=self.current_frame, color='g', linestyle=':', alpha=0.8, label='当前帧 (Current)')
                    # 高亮当前点
                    if self.current_frame < len(y):
                        curr_prob = y[self.current_frame]
                        color = 'red' if curr_prob > 0.5 else 'green'
                        self.ax_prob.plot(self.current_frame, curr_prob, 'o', color=color, markersize=6, markeredgecolor='white')
                        
                        # 显示数值标签
                        self.ax_prob.text(self.current_frame, curr_prob + 0.05, f'{curr_prob:.2f}', 
                                        ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
            
            self.ax_prob.legend(fontsize=7, loc='upper right', framealpha=0.8)
                
        except Exception as e:
            print(f"概率曲线更新错误: {e}")

    def save_high_res_plot(self):
        """保存高清图表用于论文"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="保存高清图表",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("SVG Image", "*.svg")],
                initialfile=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if file_path:
                # 临时调整分辨率和字体大小以适应出版要求
                old_size = self.fig.get_size_inches()
                self.fig.set_size_inches(16, 10) # 增大尺寸
                
                # 保存
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                
                # 恢复尺寸
                self.fig.set_size_inches(old_size)
                self.canvas.draw()
                
                messagebox.showinfo("成功", f"高清图表已保存至:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败: {str(e)}")

    def export_report(self):
        """导出检测数据报告"""
        if not hasattr(self, 'prob_history') or not self.prob_history:
            messagebox.showwarning("警告", "暂无检测数据可导出")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="导出检测报告",
                defaultextension=".csv",
                filetypes=[("CSV File", "*.csv"), ("Excel File", "*.xlsx")],
                initialfile=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if file_path:
                # 构建DataFrame
                data = {
                    'Frame': range(len(self.prob_history)),
                    'Probability': self.prob_history,
                    'Is_Nodule': [p > 0.5 for p in self.prob_history]
                }
                df = pd.DataFrame(data)
                
                # 添加元数据
                if hasattr(self, 'size_var'):
                    df['Prior_Size_mm'] = self.size_var.get()
                if hasattr(self, 'depth_var'):
                    df['Prior_Depth_mm'] = self.depth_var.get()
                
                # 保存
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)
                    
                messagebox.showinfo("成功", f"检测报告已导出至:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"导出报告失败: {str(e)}")

    def update_trend_plots_optimized(self, nodules):
        """优化的趋势图更新 - 修复维度不匹配问题"""
        try:
            # 检查是否有趋势图轴
            if not hasattr(self, 'ax_trend') or self.ax_trend is None:
                return
                
            current_time = time.time()
            
            if self.enable_elastography:
                # 弹性成像模式：统计结节概率分布
                high_prob_count = sum(1 for n in nodules if n.get('nodule_probability', 0) > 0.8)
                medium_prob_count = sum(1 for n in nodules if 0.6 < n.get('nodule_probability', 0) <= 0.8)
                low_prob_count = sum(1 for n in nodules if 0.3 < n.get('nodule_probability', 0) <= 0.6)
                
                self.trend_data['timestamps'].append(current_time)
                self.trend_data['high_prob'].append(high_prob_count)
                self.trend_data['medium_prob'].append(medium_prob_count)
                self.trend_data['low_prob'].append(low_prob_count)
                
                # 只在有足够数据时更新，并确保所有数据长度一致
                if len(self.trend_data['timestamps']) > 5:
                    self.ax_trend.clear()
                    
                    # 确保所有数据长度一致
                    min_length = min(len(self.trend_data['timestamps']), 
                                   len(self.trend_data['high_prob']),
                                   len(self.trend_data['medium_prob']),
                                   len(self.trend_data['low_prob']))
                    
                    times = list(self.trend_data['timestamps'])[-min_length:]
                    high_prob = list(self.trend_data['high_prob'])[-min_length:]
                    medium_prob = list(self.trend_data['medium_prob'])[-min_length:]
                    low_prob = list(self.trend_data['low_prob'])[-min_length:]
                    
                    # 转换为相对时间
                    if len(times) > 0:
                        relative_times = [(t - times[0]) for t in times]
                        
                        self.ax_trend.plot(relative_times, high_prob, 'r-', 
                                         linewidth=2, label='高概率(>0.8)')
                        self.ax_trend.plot(relative_times, medium_prob, 'orange', 
                                         linewidth=2, label='中概率(0.6-0.8)')
                        self.ax_trend.plot(relative_times, low_prob, 'y-', 
                                         linewidth=2, label='低概率(0.3-0.6)')
                        
                        self.ax_trend.set_title('结节概率分布趋势', fontsize=9)
                        self.ax_trend.set_xlabel('时间(s)')
                        self.ax_trend.set_ylabel('结节数量')
                        self.ax_trend.legend(fontsize=8)
                        self.ax_trend.grid(True, alpha=0.3)
            else:
                # 传统模式：结节数量趋势
                nodule_count = len(nodules)
                self.trend_data['timestamps'].append(current_time)
                self.trend_data['nodule_counts'].append(nodule_count)
                
                # 只在有足够数据时更新，并确保数据长度一致
                if len(self.trend_data['timestamps']) > 5:
                    self.ax_trend.clear()
                    
                    # 确保数据长度一致
                    min_length = min(len(self.trend_data['timestamps']), 
                                   len(self.trend_data['nodule_counts']))
                    
                    times = list(self.trend_data['timestamps'])[-min_length:]
                    counts = list(self.trend_data['nodule_counts'])[-min_length:]
                    
                    # 转换为相对时间
                    if len(times) > 0:
                        relative_times = [(t - times[0]) for t in times]
                        
                        self.ax_trend.plot(relative_times, counts, 'b-', linewidth=2)
                        self.ax_trend.set_title('结节数量趋势', fontsize=9)
                        self.ax_trend.set_xlabel('时间(s)')
                        self.ax_trend.set_ylabel('结节数量')
                        self.ax_trend.grid(True, alpha=0.3)
                    
        except Exception as e:
            print(f"趋势图更新错误: {e}")
    
    def update_statistics_optimized(self, nodules, timestamp):
        """优化的统计信息更新 - 替换为弹性成像统计"""
        try:
            if self.enable_elastography:
                # 弹性成像模式：统计结节概率分布
                self.stats_data = {'high_prob': 0, 'medium_prob': 0, 'low_prob': 0}
                
                # 计算概率分布
                for nodule in nodules:
                    prob = nodule.get('nodule_probability', 0)
                    if prob > 0.8:
                        self.stats_data['high_prob'] += 1
                    elif prob > 0.6:
                        self.stats_data['medium_prob'] += 1
                    elif prob > 0.3:
                        self.stats_data['low_prob'] += 1
                
                # 更新统计图
                if sum(self.stats_data.values()) > 0:
                    self.ax_stats.clear()
                    labels = ['高概率(>0.8)', '中概率(0.6-0.8)', '低概率(0.3-0.6)']
                    values = [self.stats_data['high_prob'], self.stats_data['medium_prob'], 
                             self.stats_data['low_prob']]
                    colors = ['red', 'orange', 'yellow']
                    
                    # 只显示非零值
                    non_zero_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
                    if non_zero_data:
                        labels, values, colors = zip(*non_zero_data)
                        self.ax_stats.pie(values, labels=labels, colors=colors, autopct='%1.0f%%',
                                        startangle=90)
                        self.ax_stats.set_title('结节概率分布', fontsize=9)
                        
                        # 添加弹性成像参数信息
                        info_text = f"弹性成像参数:\n结节原形度: {self.nodule_sphericity:.2f}\n结节尺寸: {self.nodule_size:.1f}mm"
                        self.ax_stats.text(1.2, 0.5, info_text, transform=self.ax_stats.transAxes,
                                         fontsize=8, verticalalignment='center',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            else:
                # 传统模式：风险分布统计
                self.stats_data = {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
                
                # 计算风险分布
                for nodule in nodules:
                    risk = nodule['risk_score']
                    if risk > 0.8:
                        self.stats_data['high_risk'] += 1
                    elif risk > 0.6:
                        self.stats_data['medium_risk'] += 1
                    else:
                        self.stats_data['low_risk'] += 1
                
                # 更新统计图（简化版本）
                if sum(self.stats_data.values()) > 0:
                    self.ax_stats.clear()
                    labels = ['高风险', '中风险', '低风险']
                    values = [self.stats_data['high_risk'], self.stats_data['medium_risk'], 
                             self.stats_data['low_risk']]
                    colors = ['red', 'orange', 'yellow']
                    
                    # 只显示非零值
                    non_zero_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
                    if non_zero_data:
                        labels, values, colors = zip(*non_zero_data)
                        self.ax_stats.pie(values, labels=labels, colors=colors, autopct='%1.0f%%',
                                        startangle=90)
                        self.ax_stats.set_title('风险分布', fontsize=9)
                        
        except Exception as e:
            print(f"统计更新错误: {e}")
                
        except Exception as e:
            print(f"统计更新错误: {e}")
    
    def toggle_simulation(self):
        """切换仿真播放状态"""
        if not hasattr(self, 'is_simulating'):
            self.is_simulating = False
            
        if not self.is_simulating:
            if self.data is None:
                return
            self.is_simulating = True
            self.play_btn.config(text="停止仿真")
            self.simulation_loop()
        else:
            self.is_simulating = False
            self.play_btn.config(text="播放仿真")
            
    def simulation_loop(self):
        """仿真循环"""
        if self.is_simulating and self.data is not None:
            if self.current_frame < len(self.data) - 1:
                self.current_frame += 1
                self.frame_var.set(self.current_frame)
                self.update_display()
                
                # 使用设定的更新间隔
                delay = int(self.interval_var.get())
                if delay < 10: delay = 10
                
                self.root.after(delay, self.simulation_loop)
            else:
                # 播放结束
                self.is_simulating = False
                self.play_btn.config(text="播放仿真")
                self.current_frame = 0  # 可选：循环播放或停在最后
                self.frame_var.set(self.current_frame)

    def prev_frame(self):
        """上一帧"""
        if self.data is None:
            return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_var.set(self.current_frame)
            self.update_display()
            
    def next_frame(self):
        """下一帧"""
        if self.data is None:
            return
        if self.current_frame < len(self.data) - 1:
            self.current_frame += 1
            self.frame_var.set(self.current_frame)
            self.update_display()
            
    def skip_frames_control(self, n):
        """快进/快退 n 帧"""
        if self.data is None:
            return
        new_frame = self.current_frame + n
        new_frame = max(0, min(new_frame, len(self.data) - 1))
        self.current_frame = new_frame
        self.frame_var.set(self.current_frame)
        self.update_display()
        
    def jump_to_frame(self, event=None):
        """跳转到指定帧"""
        if self.data is None:
            return
        try:
            target = int(self.jump_var.get())
            if 0 <= target < len(self.data):
                self.current_frame = target
                self.frame_var.set(target)
                self.update_display()
            else:
                messagebox.showwarning("警告", f"帧号超出范围 (0-{len(self.data)-1})")
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的帧号数字")

    def load_csv_file(self):
        """加载CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 使用pandas读取CSV
                df = pd.read_csv(file_path)
                
                # 智能识别数据列
                # 1. 尝试查找以MAT_开头的列
                mat_cols = [c for c in df.columns if str(c).strip().startswith('MAT_')]
                
                if mat_cols:
                    # 按数字后缀排序 (MAT_0, MAT_1, ...)
                    try:
                        mat_cols.sort(key=lambda x: int(str(x).strip().split('_')[1]))
                        self.data = df[mat_cols].values
                    except:
                        # 如果排序失败，直接使用找到的列
                        self.data = df[mat_cols].values
                else:
                    # 2. 如果没有MAT_列，尝试直接使用所有数据
                    self.data = df.values
                
                # 验证数据维度
                if self.data.shape[1] != 96:
                    # 如果列数不是96，尝试智能修复
                    if self.data.shape[1] > 96:
                        # 假设最后96列是传感器数据
                        self.data = self.data[:, -96:]
                    else:
                        messagebox.showwarning("数据警告", f"数据列数({self.data.shape[1]})不符合预期(96)，可能导致显示错误")

                self.current_frame = 0
                self.frame_scale.config(to=len(self.data)-1)
                self.frame_var.set(0)
                
                # 启用播放按钮
                if hasattr(self, 'play_btn'):
                    self.play_btn.config(state="normal")
                
                self.update_display()
                self.status_var.set(f"已加载 {len(self.data)} 帧数据")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    
    def on_frame_change(self, value):
        """帧变化回调"""
        self.current_frame = int(float(value))
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.data is None:
            return
        
        try:
            frame_data = self.data[self.current_frame]
            matrix = frame_data.reshape(12, 8)
            
            # AI深度学习检测 (针对文件回放)
            if self.use_dl_detection and self.dl_var.get() and self.dl_model:
                try:
                    # 准备序列数据 (10帧)
                    if self.current_frame >= 9:
                        seq_raw = self.data[self.current_frame-9 : self.current_frame+1]
                    else:
                        # 填充不足的帧
                        pad_len = 10 - (self.current_frame + 1)
                        first_frame = self.data[[0]]
                        pad = np.repeat(first_frame, pad_len, axis=0)
                        current_slice = self.data[0 : self.current_frame+1]
                        seq_raw = np.vstack([pad, current_slice])
                    
                    # 预处理数据
                    # Reshape to (10, 12, 8)
                    seq_raw_reshaped = []
                    for i in range(10):
                        # Ensure 96 elements
                        fr = seq_raw[i]
                        if fr.size > 96: fr = fr[-96:]
                        seq_raw_reshaped.append(fr.reshape(12, 8))
                    seq_raw_reshaped = np.array(seq_raw_reshaped)
                    
                    # Intensity Stats
                    avg_intensity = np.mean(seq_raw_reshaped)
                    max_intensity = np.max(seq_raw_reshaped)
                    std_intensity = np.std(seq_raw_reshaped)
                    intensity_stats = torch.tensor([[avg_intensity, max_intensity, std_intensity]], dtype=torch.float32).to(self.device)
                    
                    # Normalize Sequence
                    seq_norm = np.zeros((10, 1, 12, 8), dtype=np.float32)
                    for i in range(10):
                        fr = seq_raw_reshaped[i]
                        mn, mx = fr.min(), fr.max()
                        if mx - mn > 1e-6:
                            fr = (fr - mn) / (mx - mn)
                        else:
                            fr = fr - mn
                        seq_norm[i, 0] = fr
                        
                    # To Tensor: (1, Seq, 1, 12, 8)
                    input_tensor = torch.tensor(seq_norm).unsqueeze(0).to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        prob, size, depth = self.dl_model(input_tensor, intensity_stats)
                        self.dl_probability = prob.item()
                        self.dl_pred_size = size.item()
                        self.dl_pred_depth = depth.item()
                except Exception as e:
                    print(f"文件回放AI推理错误: {e}")
                    self.dl_probability = 0.0
                
                # 记录概率用于曲线显示
                if not hasattr(self, 'prob_history'):
                    self.prob_history = []
                
                # 暂时只做实时追加，为了平滑显示
                if len(self.prob_history) > self.current_frame:
                    self.prob_history[self.current_frame] = self.dl_probability
                else:
                    self.prob_history.append(self.dl_probability)
                
                self.update_prob_curve()
            else:
                self.dl_probability = 0.0
            
            # 执行检测
            if self.use_enhanced_detection and self.enhanced_detector.is_trained:
                # 使用增强检测系统
                enhanced_result = self.enhanced_detector.process_frame(frame_data, self.current_frame)
                if enhanced_result:
                    normalized = enhanced_result['matrix_data']['normalized_matrix']
                    prob_map = np.ones_like(normalized) * enhanced_result['combined_probability']
                    
                    # 创建简化的结节列表
                    nodules = []
                    if enhanced_result['combined_probability'] > 0.5:
                        center = np.unravel_index(np.argmax(normalized), normalized.shape)
                        nodules.append({
                            'centroid': center,
                            'risk_score': enhanced_result['combined_probability'],
                            'area': 1.0,
                            'enhanced': True
                        })
                    
                    nodule_mask = prob_map > 0.5
                else:
                    # 回退到标准检测
                    normalized, nodule_mask, nodules = self.detector.advanced_nodule_detection(
                        matrix, self.current_frame
                    )
                    prob_map = normalized
            else:
                # 使用标准检测
                normalized, nodule_mask, nodules = self.detector.advanced_nodule_detection(
                    matrix, self.current_frame
                )
                prob_map = normalized  # 使用normalized作为prob_map
            
            # 更新可视化
            self.update_realtime_plots_optimized(normalized, nodule_mask, nodules, prob_map)
            
            # 更新帧标签
            self.frame_label.config(text=f"{self.current_frame}/{len(self.data)-1}")
            
        except Exception as e:
            print(f"显示更新错误: {e}")
    
    def on_detection_mode_change(self, event=None):
        """检测模式变化回调"""
        mode = self.detection_mode_var.get()
        if mode == "增强应力":
            self.use_enhanced_detection = True
            self.status_var.set("已切换到增强应力检测模式")
        else:
            self.use_enhanced_detection = False
            self.status_var.set("已切换到标准检测模式")
    
    def show_training_dialog(self):
        """显示训练数据输入对话框"""
        if not self.use_enhanced_detection:
            messagebox.showinfo("提示", "请先切换到增强应力检测模式")
            return
        
        # 创建训练数据输入窗口
        training_window = tk.Toplevel(self.root)
        training_window.title("添加训练数据")
        training_window.geometry("400x300")
        training_window.transient(self.root)
        training_window.grab_set()
        
        # 数据类型选择
        ttk.Label(training_window, text="数据类型:").pack(pady=5)
        data_type_var = tk.StringVar(value="结节")
        data_type_frame = ttk.Frame(training_window)
        data_type_frame.pack(pady=5)
        ttk.Radiobutton(data_type_frame, text="结节", variable=data_type_var, value="结节").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(data_type_frame, text="正常", variable=data_type_var, value="正常").pack(side=tk.LEFT, padx=10)
        
        # 结节参数输入框架
        params_frame = ttk.LabelFrame(training_window, text="结节参数", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 面积
        ttk.Label(params_frame, text="面积 (cm²):").grid(row=0, column=0, sticky=tk.W, pady=2)
        area_var = tk.DoubleVar(value=2.0)
        ttk.Entry(params_frame, textvariable=area_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        # 直径
        ttk.Label(params_frame, text="直径 (cm):").grid(row=1, column=0, sticky=tk.W, pady=2)
        diameter_var = tk.DoubleVar(value=1.5)
        ttk.Entry(params_frame, textvariable=diameter_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # 深度
        ttk.Label(params_frame, text="深度 (cm):").grid(row=2, column=0, sticky=tk.W, pady=2)
        depth_var = tk.DoubleVar(value=0.8)
        ttk.Entry(params_frame, textvariable=depth_var, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        # 位置
        ttk.Label(params_frame, text="位置 (x,y):").grid(row=3, column=0, sticky=tk.W, pady=2)
        position_frame = ttk.Frame(params_frame)
        position_frame.grid(row=3, column=1, padx=5, pady=2)
        x_var = tk.IntVar(value=6)
        y_var = tk.IntVar(value=4)
        ttk.Entry(position_frame, textvariable=x_var, width=6).pack(side=tk.LEFT)
        ttk.Label(position_frame, text=",").pack(side=tk.LEFT)
        ttk.Entry(position_frame, textvariable=y_var, width=6).pack(side=tk.LEFT)
        
        def add_current_data():
            """添加当前数据作为训练样本"""
            if self.data is None:
                messagebox.showwarning("警告", "没有当前数据，请先加载数据或连接传感器")
                return
            
            try:
                # 获取当前帧数据
                if hasattr(self, 'current_frame') and self.data is not None:
                    current_data = self.data[self.current_frame]
                else:
                    # 使用模拟数据
                    current_data = np.random.normal(0.5, 0.1, 96)
                
                is_nodule = data_type_var.get() == "结节"
                
                if is_nodule:
                    success = self.enhanced_detector.add_training_data(
                        current_data,
                        area=area_var.get(),
                        diameter=diameter_var.get(),
                        depth=depth_var.get(),
                        position=(x_var.get(), y_var.get()),
                        is_nodule=True
                    )
                else:
                    success = self.enhanced_detector.add_training_data(
                        current_data,
                        is_nodule=False
                    )
                
                if success:
                    messagebox.showinfo("成功", "训练数据已添加")
                    training_window.destroy()
                else:
                    messagebox.showerror("错误", "添加训练数据失败")
                    
            except Exception as e:
                messagebox.showerror("错误", f"添加训练数据时出错: {str(e)}")
        
        # 按钮
        button_frame = ttk.Frame(training_window)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="添加数据", command=add_current_data).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="取消", command=training_window.destroy).pack(side=tk.LEFT, padx=10)
    
    def train_enhanced_system(self):
        """训练增强检测系统"""
        if not self.use_enhanced_detection:
            messagebox.showinfo("提示", "请先切换到增强应力检测模式")
            return
        
        try:
            self.status_var.set("正在训练系统...")
            self.root.update()
            
            # 在后台线程中训练
            def train_thread():
                success = self.enhanced_detector.train_system()
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.on_training_complete(success))
            
            threading.Thread(target=train_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("错误", f"训练系统时出错: {str(e)}")
            self.status_var.set("训练失败")
    
    def on_training_complete(self, success):
        """训练完成回调"""
        if success:
            messagebox.showinfo("成功", "系统训练完成！")
            self.status_var.set("系统训练完成，可以开始增强检测")
        else:
            messagebox.showerror("失败", "系统训练失败，请检查训练数据")
    
    def add_sensor_grid_overlay(self, ax, data):
        """在图表上添加传感器网格叠加"""
        try:
            rows, cols = data.shape
            
            # 绘制网格线
            for i in range(rows + 1):
                ax.axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
            for j in range(cols + 1):
                ax.axvline(x=j-0.5, color='white', linewidth=0.5, alpha=0.3)
            
            # 在每个传感器位置添加数值标签（仅在缩放时显示）
            if hasattr(self, 'zoom_factor') and self.zoom_factor > 1.5:
                for i in range(rows):
                    for j in range(cols):
                        value = data[i, j]
                        ax.text(j, i, f'{value:.1f}', 
                               ha='center', va='center', 
                               fontsize=6, color='white',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='black', alpha=0.5))
        except Exception as e:
            print(f"传感器网格叠加错误: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizedDetectionGUI(root)
    root.mainloop()
