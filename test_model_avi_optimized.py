#!/usr/bin/env python3
"""
YOLO8 Interactive GUI Detector - OPTIMIZED VERSION
FPS-optimized version with input resizing, single or multiple object detection, and other improvements
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import os
import sys
import threading
import queue
import time
import json
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np

# Try to import YOLO, show error if not available
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Error: ultralytics or torch not installed. Run: pip install ultralytics torch")
    sys.exit(1)

class YOLOTubeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Tube v0.7.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Config file path
        self.config_file = "config.json"
        
        # Detection variables - Optimized for object detection
        self.model = None
        self.model_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.confidence = tk.DoubleVar(value=0.35)
        self.iou_threshold = tk.DoubleVar(value=0.4)
        self.is_detecting = False
        self.is_paused = False
        self.detection_thread = None
        self.video_capture = None
        self.video_capture_lock = threading.Lock()  # Thread safety for video capture operations
        self.video_position = 0  # Track video position for pause/resume
        self.video_total_frames = 0  # Total frames in video for progress bar
        self.video_fps = 30  # Video FPS for time calculation
        self.last_seek_frame = -1  # Track last seek frame to prevent log spam
        self.last_seek_time = 0  # Track last seek time for debouncing
        self.seek_logging_enabled = True  # Enable/disable seek logging
        self.detection_log_entries = {}  # Track detection log entries for click handling
        self.seek_initialized = False  # Track if seek is initialized to prevent initial trigger
        
        # FPS OPTIMIZATION VARIABLES
        self.input_size = tk.StringVar(value="Full Square")  # Resize input for faster processing (can be "Full Square", "Full Screen", or numeric string)
        self.max_detections = tk.IntVar(value=1)  # Limit detections per frame
        self.single_object_mode = tk.BooleanVar(value=True)  # Detect only one object
        self.target_class = tk.StringVar(value="")  # Target class for single object mode (will be set when model loads)
        self.enable_tracking = tk.BooleanVar(value=False)  # Enable object tracking (disabled for max speed)
        self.use_grayscale = tk.BooleanVar(value=False)  # Convert detection area to grayscale for faster processing
        
        # GPU ACCELERATION VARIABLES
        self.use_gpu = tk.BooleanVar(value=True)  # Enable GPU acceleration
        self.gpu_device = tk.StringVar()  # GPU device selection (will be set after getting devices)
        self.mixed_precision = tk.BooleanVar(value=True)  # Use FP16 for faster inference
        self.gpu_memory_fraction = tk.DoubleVar(value=0.8)  # GPU memory usage (80% default)
        self.tracked_objects = {}  # Store tracked objects
        self.track_id_counter = 0
        
        # Detection area tracking
        self.detection_area_center = None  # Current center of detection area
        self.last_detection_center = None  # Center of last detected object
        self.detection_area_size = None    # Current detection area size
        self.frames_since_last_detection = 0  # Counter for when object is lost
        
        # Dynamic model-dependent settings
        self.model_classes = []  # Will be populated when model is loaded
        self.model_colors = {}   # Will be populated when model is loaded
        
        # Statistics
        self.total_frames = 0
        self.detection_count = 0
        self.current_fps = 0
        self.processing_times = []
        
        # GUI queues for thread communication
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self.stats_queue = queue.Queue()
        
        # Colors for detection boxes
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        # Load saved configuration
        self.load_config()
        
        self.setup_gui()
        
        # Set default GPU device
        gpu_devices = self.get_available_gpu_devices()
        if gpu_devices and gpu_devices[0] != "CPU Only":
            self.gpu_device.set(gpu_devices[0])  # Set to first available GPU
        
        self.update_display()
    
    def generate_model_colors(self, class_names):
        """Generate colors for model classes"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 128), (128, 128, 0), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (255, 128, 0), (128, 255, 0), (0, 255, 128)
        ]
        
        model_colors = {}
        for i, class_name in enumerate(class_names):
            model_colors[class_name.lower()] = colors[i % len(colors)]
        
        return model_colors
    
    def update_target_class_combobox(self):
        """Update the target class combobox with current model classes"""
        if hasattr(self, 'target_class_combo'):
            self.target_class_combo['values'] = self.model_classes
    
    def setup_gui(self):
        """Setup the GUI interface with optimization controls"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=3)  # Video row (more space)
        main_frame.rowconfigure(2, weight=1)  # Statistics row (less space)
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLO Tube v0.7.0", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Configuration & Optimization", padding="10")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.setup_controls(controls_frame)
        
        # Right panel - Video display (now takes more space)
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.setup_video_display(video_frame)
        
        # Bottom panel - Statistics (moved underneath)
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics & Performance", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.setup_statistics(stats_frame)
    
    def setup_controls(self, parent):
        """Setup control panel with optimization options"""
        row = 0
        
        # Model selection
        ttk.Label(parent, text="YOLO8 Model:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        model_frame = ttk.Frame(parent)
        model_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(0, weight=1)
        
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=40)
        self.model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(model_frame, text="Browse", command=self.select_model).grid(row=0, column=1)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=2, padx=(5, 0))
        row += 1
        
        # Model info display
        self.model_info = scrolledtext.ScrolledText(parent, height=4, width=50, font=('Courier', 9))
        self.model_info.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.model_info.insert(tk.END, "📝 Model Information:\nNo model loaded yet...")
        self.model_info.config(state=tk.DISABLED)
        row += 1
        
        # Video source selection
        ttk.Label(parent, text="Video Source:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
        video_frame = ttk.Frame(parent)
        video_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        
        self.video_entry = ttk.Entry(video_frame, textvariable=self.video_path, width=40)
        self.video_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(video_frame, text="Browse", command=self.select_video).grid(row=0, column=1)
        ttk.Button(video_frame, text="Webcam", command=self.use_webcam).grid(row=0, column=2, padx=(5, 0))
        row += 1
        
        # FPS OPTIMIZATION CONTROLS
        ttk.Label(parent, text="🚀 FPS Optimization:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
        # Detection area size optimization
        size_frame = ttk.Frame(parent)
        size_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(size_frame, text="Detection Area:").grid(row=0, column=0, sticky=tk.W)
        size_combo = ttk.Combobox(size_frame, textvariable=self.input_size, 
                                 values=["Full Square", "Full Screen", "416", "512", "640", "832", "1024"], width=10)
        size_combo.grid(row=0, column=1, padx=5)
        ttk.Label(size_frame, text="(Full Screen = no square)").grid(row=0, column=2, sticky=tk.W)
        row += 1
        
        # Single object mode
        single_frame = ttk.Frame(parent)
        single_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Checkbutton(single_frame, text="Single Object Mode", 
                       variable=self.single_object_mode).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(single_frame, text="Target:").grid(row=0, column=1, padx=(10, 5))
        self.target_class_combo = ttk.Combobox(single_frame, textvariable=self.target_class, 
                                              values=["Load model first..."], width=12)
        self.target_class_combo.grid(row=0, column=2)
        row += 1
        
        # Max detections limit
        max_frame = ttk.Frame(parent)
        max_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(max_frame, text="Max Detections:").grid(row=0, column=0, sticky=tk.W)
        max_scale = ttk.Scale(max_frame, from_=1, to=50, variable=self.max_detections, 
                             orient=tk.HORIZONTAL, length=150)
        max_scale.grid(row=0, column=1, padx=5)
        self.max_label = ttk.Label(max_frame, text="1")
        self.max_label.grid(row=0, column=2)
        max_scale.configure(command=self.update_max_label)
        row += 1
        
        # Grayscale optimization (experimental)
        gray_frame = ttk.Frame(parent)
        gray_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Checkbutton(gray_frame, text="Use Grayscale Detection", 
                       variable=self.use_grayscale, command=self.on_grayscale_toggle).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(gray_frame, text="(YOLO expects 3-channel input - may cause errors)").grid(row=0, column=1, padx=(10, 5))
        row += 1
        
        # Object tracking
        track_frame = ttk.Frame(parent)
        track_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Checkbutton(track_frame, text="Enable Object ID Tracking", 
                       variable=self.enable_tracking).grid(row=0, column=0, sticky=tk.W)
        row += 1
        
        # GPU ACCELERATION CONTROLS
        ttk.Label(parent, text="🚀 GPU Acceleration:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
                # GPU enable/disable
        gpu_frame = ttk.Frame(parent)
        gpu_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Checkbutton(gpu_frame, text="Enable GPU Acceleration", 
                       variable=self.use_gpu).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(gpu_frame, text="Device:").grid(row=0, column=1, padx=(10, 5))
        
        # Get available GPU devices
        gpu_devices = self.get_available_gpu_devices()
        gpu_combo = ttk.Combobox(gpu_frame, textvariable=self.gpu_device, 
                                 values=gpu_devices, width=15)
        gpu_combo.grid(row=0, column=2)
        row += 1
        
        # Mixed precision
        precision_frame = ttk.Frame(parent)
        precision_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Checkbutton(precision_frame, text="Mixed Precision (FP16)", 
                       variable=self.mixed_precision).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(precision_frame, text="(2x faster)").grid(row=0, column=1, padx=(10, 0))
        row += 1
        
        # GPU memory usage
        memory_frame = ttk.Frame(parent)
        memory_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(memory_frame, text="GPU Memory:").grid(row=0, column=0, sticky=tk.W)
        memory_scale = ttk.Scale(memory_frame, from_=0.3, to=0.95, variable=self.gpu_memory_fraction, 
                                orient=tk.HORIZONTAL, length=150)
        memory_scale.grid(row=0, column=1, padx=5)
        self.memory_label = ttk.Label(memory_frame, text="80%")
        self.memory_label.grid(row=0, column=2)
        memory_scale.configure(command=self.update_memory_label)
        row += 1
        
        # Detection parameters
        ttk.Label(parent, text="Detection Parameters:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5))
        row += 1
        
        # Confidence threshold
        conf_frame = ttk.Frame(parent)
        conf_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(conf_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        conf_scale = ttk.Scale(conf_frame, from_=0.2, to=0.8, variable=self.confidence, 
                              orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=0, column=1, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.35")
        self.conf_label.grid(row=0, column=2)
        conf_scale.configure(command=self.update_conf_label)
        row += 1
        
        # IoU threshold
        iou_frame = ttk.Frame(parent)
        iou_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(iou_frame, text="IoU Threshold:").grid(row=0, column=0, sticky=tk.W)
        iou_scale = ttk.Scale(iou_frame, from_=0.2, to=0.7, variable=self.iou_threshold, 
                             orient=tk.HORIZONTAL, length=200)
        iou_scale.grid(row=0, column=1, padx=5)
        self.iou_label = ttk.Label(iou_frame, text="0.4")
        self.iou_label.grid(row=0, column=2)
        iou_scale.configure(command=self.update_iou_label)
        row += 1
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, pady=20)
        
        ttk.Button(button_frame, text="📷 Save Frame", 
                  command=self.save_current_frame).grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="🔄 Reset Stats", 
                  command=self.reset_statistics).grid(row=0, column=1, padx=5)
    
    def setup_video_display(self, parent):
        """Setup video display area"""
        # Video canvas
        self.video_canvas = tk.Canvas(parent, bg='black', width=640, height=480)
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Bind click event to pause/unpause detection
        self.video_canvas.bind("<Button-1>", self.on_video_canvas_click)
        
        # Default message
        self.video_canvas.create_text(320, 240, text="No video loaded\nSelect model and video source\n\nClick video area to pause/unpause during detection", 
                                     fill='white', font=('Arial', 14), justify=tk.CENTER)
        
        # Video progress bar frame
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        progress_frame.columnconfigure(1, weight=1)
        
        # Time labels
        self.current_time_label = ttk.Label(progress_frame, text="00:00", font=('Arial', 9))
        self.current_time_label.grid(row=0, column=0, padx=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     command=self.seek_video, state=tk.DISABLED)
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Bind mouse release event to resume playback
        self.progress_bar.bind("<ButtonRelease-1>", self.on_seek_release)
        
        self.total_time_label = ttk.Label(progress_frame, text="00:00", font=('Arial', 9))
        self.total_time_label.grid(row=0, column=2, padx=(5, 0))
        
        # Video controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Main control button (moved here from left panel)
        self.main_control_button = ttk.Button(controls_frame, text="🔴 Start Detection & Video", 
                                             command=self.toggle_detection)
        self.main_control_button.grid(row=0, column=0, padx=5)
        
        # Frame navigation buttons
        self.frame_back_button = ttk.Button(controls_frame, text="⏪ -1 Frame", 
                                           command=self.frame_backward, state=tk.DISABLED)
        self.frame_back_button.grid(row=0, column=1, padx=5)
        
        self.frame_forward_button = ttk.Button(controls_frame, text="+1 Frame ⏩", 
                                              command=self.frame_forward, state=tk.DISABLED)
        self.frame_forward_button.grid(row=0, column=2, padx=5)
        
        # Frame info
        self.frame_info_label = ttk.Label(controls_frame, text="Frame: 0 / 0", font=('Arial', 9))
        self.frame_info_label.grid(row=0, column=3, padx=20)
    
    def setup_statistics(self, parent):
        """Setup statistics panel with performance metrics"""
        # Configure parent grid weights for horizontal layout
        parent.columnconfigure(0, weight=0)  # Statistics column (fixed width)
        parent.columnconfigure(1, weight=1)  # Log column (expandable for wider display)
        parent.rowconfigure(0, weight=1)     # Main content row
        
        # Left side - Statistics display
        stats_frame = ttk.Frame(parent)
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 10))
        
        # Statistics labels - Enhanced with performance metrics
        self.stats_labels = {}
        stats_info = [
            ("Model Status", "Ready for dronemosel-yolo8-best.pt"),
            ("Video Status", "Not loaded"),
            ("Device", "CPU"),
            ("Total Frames", "0"),
            ("Object Detections", "0"),
            ("Current FPS", "0.0"),
            ("Avg Processing Time", "0.0ms"),
            ("Detection Area", "Full Square"),
            ("GPU Memory", "N/A"),
            ("Tracking Objects", "0")
        ]
        
        for i, (label, value) in enumerate(stats_info):
            ttk.Label(stats_frame, text=f"{label}:", font=('Arial', 9, 'bold')).grid(
                row=i, column=0, sticky=tk.W, padx=(0, 10))
            self.stats_labels[label] = ttk.Label(stats_frame, text=value, font=('Arial', 9))
            self.stats_labels[label].grid(row=i, column=1, sticky=tk.W)
        
        # Right side - Log area
        log_frame = ttk.Frame(parent)
        log_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        ttk.Label(log_frame, text="Performance Log:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=70, font=('Courier', 8))
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Enable click events for detection log entries
        self.log_text.bind("<Button-1>", self.on_log_click)
        self.log_text.bind("<Motion>", self.on_log_hover)
        self.log_text.tag_configure("detection_link", foreground="blue", underline=True)
        self.log_text.tag_configure("detection_link_hover", foreground="red", underline=True)
        self.log_text.tag_configure("bold_text", font=('Courier', 8, 'bold'))
        
        self.log_text.insert(tk.END, "🚀 Performance Log:\n" + "="*30 + "\n")
        self.log_text.config(state=tk.DISABLED)
        
        # Export log button
        ttk.Button(log_frame, text="💾 Export Log", command=self.export_log).grid(
            row=2, column=0, pady=(5, 0), sticky=tk.W)
    
    def update_conf_label(self, value):
        """Update confidence label"""
        self.conf_label.config(text=f"{float(value):.2f}")
    
    def update_iou_label(self, value):
        """Update IoU label"""
        self.iou_label.config(text=f"{float(value):.2f}")
    
    def update_max_label(self, value):
        """Update max detections label"""
        self.max_label.config(text=str(int(float(value))))
    
    def update_memory_label(self, value):
        """Update GPU memory usage label"""
        percentage = int(float(value) * 100)
        self.memory_label.config(text=f"{percentage}%")
    
    def on_grayscale_toggle(self):
        """Handle grayscale checkbox toggle"""
        if self.use_grayscale.get():
            self.log_message("⚠️ Grayscale detection enabled - YOLO models expect 3-channel input and may produce errors")
        else:
            self.log_message("✅ Grayscale detection disabled - using standard 3-channel input")
    
    def get_available_gpu_devices(self):
        """Get list of available GPU devices with names"""
        devices = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    # Shorten common GPU names
                    if "4060" in gpu_name:
                        gpu_name = "RTX 4060"
                    elif "4070" in gpu_name:
                        gpu_name = "RTX 4070"
                    elif "4080" in gpu_name:
                        gpu_name = "RTX 4080"
                    elif "4090" in gpu_name:
                        gpu_name = "RTX 4090"
                    elif "3060" in gpu_name:
                        gpu_name = "RTX 3060"
                    elif "3070" in gpu_name:
                        gpu_name = "RTX 3070"
                    elif "3080" in gpu_name:
                        gpu_name = "RTX 3080"
                    elif "3090" in gpu_name:
                        gpu_name = "RTX 3090"
                    elif "A100" in gpu_name:
                        gpu_name = "A100"
                    elif "V100" in gpu_name:
                        gpu_name = "V100"
                    elif "TITAN" in gpu_name:
                        gpu_name = "TITAN"
                    
                    devices.append(f"{gpu_name} (GPU {i})")
                return devices
            else:
                return ["CPU Only"]
        except:
            return ["CPU Only"]
    
    def select_model(self):
        """Select YOLO8 model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO8 Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir=os.path.expanduser("~")
        )
        if file_path:
            self.model_path.set(file_path)
            self.save_config()  # Auto-save when model is selected
    
    def select_video(self):
        """Select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All Files", "*.*")
            ],
            initialdir=os.path.expanduser("~")
        )
        if file_path:
            self.video_path.set(file_path)
            self.save_config()  # Auto-save when video is selected
    
    def use_webcam(self):
        """Set webcam as video source"""
        self.video_path.set("0")
        self.save_config()  # Auto-save when webcam is selected
    
    def load_model(self):
        """Load the selected YOLO8 model"""
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file first!")
            return
        
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", f"Model file not found: {self.model_path.get()}")
            return
        
        try:
            self.log_message("📦 Loading model...")
            
            # GPU acceleration setup
            device = "cpu"
            if self.use_gpu.get():
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Extract GPU ID from device selection
                        device_selection = self.gpu_device.get()
                        if "GPU" in device_selection:
                            gpu_id = device_selection.split("GPU ")[1].split(")")[0]
                        else:
                            gpu_id = "0"  # Default to first GPU
                        
                        device = f"cuda:{gpu_id}"
                        self.log_message(f"🚀 GPU acceleration enabled: {device}")
                        
                        # Set GPU memory fraction
                        memory_fraction = self.gpu_memory_fraction.get()
                        torch.cuda.set_per_process_memory_fraction(memory_fraction)
                        self.log_message(f"💾 GPU memory limit: {memory_fraction*100:.0f}%")
                        
                        # Enable mixed precision if requested
                        if self.mixed_precision.get():
                            self.log_message("⚡ Mixed precision (FP16) enabled")
                    else:
                        self.log_message("⚠️ CUDA not available, using CPU")
                        device = "cpu"
                except Exception as e:
                    self.log_message(f"⚠️ GPU setup failed: {e}, using CPU")
                    device = "cpu"
            else:
                self.log_message("🖥️ Using CPU mode")
            
            # Load model with device specification
            self.model = YOLO(self.model_path.get())
            
            # Move model to specified device
            if device != "cpu":
                self.model.to(device)
            
            # Update model info
            self.model_info.config(state=tk.NORMAL)
            self.model_info.delete(1.0, tk.END)
            
            # Get device name for display
            device_name = "CPU"
            if device != "cpu":
                try:
                    import torch
                    # Extract GPU ID from device selection
                    device_selection = self.gpu_device.get()
                    if "GPU" in device_selection:
                        device_id = int(device_selection.split("GPU ")[1].split(")")[0])
                    else:
                        device_id = 0
                    
                    device_name = torch.cuda.get_device_name(device_id)
                    # Shorten common GPU names
                    if "4060" in device_name:
                        device_name = "RTX 4060"
                    elif "4070" in device_name:
                        device_name = "RTX 4070"
                    elif "4080" in device_name:
                        device_name = "RTX 4080"
                    elif "4090" in device_name:
                        device_name = "RTX 4090"
                    elif "3060" in device_name:
                        device_name = "RTX 3060"
                    elif "3070" in device_name:
                        device_name = "RTX 3070"
                    elif "3080" in device_name:
                        device_name = "RTX 3080"
                    elif "3090" in device_name:
                        device_name = "RTX 3090"
                except:
                    device_name = device
            
            model_info = f"📝 Optimized Model Information:\n"
            model_info += f"{'='*40}\n"
            model_info += f"File: {Path(self.model_path.get()).name}\n"
            model_info += f"Task: {self.model.task}\n"
            model_info += f"Device: {device_name}\n"
            model_info += f"GPU Memory: {self.gpu_memory_fraction.get()*100:.0f}%\n"
            model_info += f"Mixed Precision: {'Yes' if self.mixed_precision.get() and device != 'cpu' else 'No'}\n"
            # Calculate detection area size for display
            input_size_str = self.input_size.get()
            if input_size_str == "Full Square":
                model_info += f"Detection Area: Full Square (Auto-calculated)\n"
            elif input_size_str == "Full Screen":
                model_info += f"Detection Area: Full Screen (No Square Constraint)\n"
            else:
                model_info += f"Detection Area: {input_size_str}x{input_size_str}\n"
            model_info += f"Single Object: {'Yes' if self.single_object_mode.get() else 'No'}\n"
            model_info += f"Max Detections: {self.max_detections.get()}\n"
            model_info += f"Tracking: {'Enabled' if self.enable_tracking.get() else 'Disabled'}\n"
            
            if hasattr(self.model, 'names'):
                classes = list(self.model.names.values())
                model_info += f"Classes ({len(classes)}): {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}\n"
                
                # Populate dynamic model classes and colors
                self.model_classes = classes
                self.model_colors = self.generate_model_colors(classes)
                
                # Update target class combobox
                self.update_target_class_combobox()
                
                # Set default target class to first class if none selected
                if classes:
                    if not self.target_class.get() or self.target_class.get() not in classes:
                        self.target_class.set(classes[0])
                else:
                    self.target_class.set("")
                    self.log_message("⚠️ Warning: Model has no class information")
            else:
                model_info += "Classes: No class information available\n"
                self.model_classes = []
                self.model_colors = {}
            
            # Get model size
            file_size = os.path.getsize(self.model_path.get()) / (1024*1024)
            model_info += f"Size: {file_size:.1f} MB\n"
            model_info += f"Status: ✅ Ready for optimized detection"
            
            self.model_info.insert(tk.END, model_info)
            self.model_info.config(state=tk.DISABLED)
            
            self.stats_labels["Model Status"].config(text="✅ Loaded", foreground="green")
            self.stats_labels["Device"].config(text=device_name, foreground="green" if device != "cpu" else "blue")
            self.log_message("✅ Model loaded successfully!")
            
        except Exception as e:
            self.model_info.config(state=tk.NORMAL)
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, f"❌ Error loading model:\n{str(e)}")
            self.model_info.config(state=tk.DISABLED)
            
            self.stats_labels["Model Status"].config(text="❌ Error", foreground="red")
            self.stats_labels["Device"].config(text="CPU", foreground="red")
            self.log_message(f"❌ Model loading failed: {e}")
            messagebox.showerror("Model Loading Error", str(e))
    
    def toggle_detection(self):
        """Smart toggle: Start detection, or pause/resume if already running"""
        if not self.is_detecting:
            self.start_detection()
        elif self.is_paused:
            self.resume_detection()
        else:
            self.pause_detection()
    

    
    def pause_detection(self):
        """Pause video detection"""
        if self.is_detecting and not self.is_paused:
            self.is_paused = True
            self.main_control_button.config(text="▶️ Resume Detection")
            self.stats_labels["Video Status"].config(text="⏸️ Paused", foreground="yellow")
            self.log_message("⏸️ Detection paused")
    
    def resume_detection(self):
        """Resume video detection"""
        if self.is_detecting and self.is_paused:
            self.is_paused = False
            self.main_control_button.config(text="⏸️ Pause Detection")
            self.stats_labels["Video Status"].config(text="🔴 Detecting", foreground="red")
            
            # Ensure video capture is positioned correctly when resuming
            if self.video_capture and not isinstance(self.video_path.get(), int):
                with self.video_capture_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames)
            
            self.log_message("▶️ Detection resumed - continuing from frame " + str(self.total_frames))
    
    def start_detection(self):
        """Start video detection"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video source!")
            return
        
        # Check video source
        video_source = self.video_path.get()
        if video_source == "0":
            video_source = 0
        elif not os.path.exists(video_source):
            messagebox.showerror("Error", f"Video file not found: {video_source}")
            return
        
        self.is_detecting = True
        self.is_paused = False
        self.main_control_button.config(text="⏸️ Pause Detection")
        self.frame_back_button.config(state=tk.NORMAL)
        self.frame_forward_button.config(state=tk.NORMAL)
        self.stats_labels["Video Status"].config(text="🔴 Detecting", foreground="red")
        
        # Reset tracking
        self.tracked_objects = {}
        self.track_id_counter = 0
        self.last_seek_frame = -1  # Reset seek frame tracking
        self.last_seek_time = 0  # Reset seek time tracking
        self.detection_log_entries = {}  # Clear detection log entries
        self.seek_initialized = False  # Reset seek initialization flag
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_worker, 
                                                args=(video_source,), daemon=True)
        self.detection_thread.start()
        
        self.log_message("🚀 Optimized detection started!")
    

    
    def seek_video(self, value):
        """Seek to specific position in video with frame preview"""
        if self.is_detecting and self.video_capture and not isinstance(self.video_path.get(), int) and self.seek_initialized:
            try:
                # Convert percentage to frame number
                target_frame = int((float(value) / 100.0) * self.video_total_frames)
                
                # Debounce: Only process if frame changed and enough time has passed
                current_time = time.time()
                if (target_frame != self.last_seek_frame and 
                    current_time - self.last_seek_time > 0.1):  # 100ms debounce
                    self.last_seek_frame = target_frame
                    self.last_seek_time = current_time
                    
                    # Pause detection
                    if not self.is_paused:
                        self.pause_detection()
                    
                    # Run detection on this frame (same as normal detection)
                    self.run_detection_on_frame_simple(target_frame)
                    
                    if self.seek_logging_enabled:
                        self.log_message(f"⏩ Seeking to frame {target_frame}")
            except Exception as e:
                self.log_message(f"❌ Seek error: {e}")
    
    def frame_backward(self):
        """Move one frame backward and run detection"""
        if self.is_detecting and self.video_capture and not isinstance(self.video_path.get(), int):
            try:
                target_frame = max(0, self.total_frames - 1)
                
                # Pause detection
                if not self.is_paused:
                    self.pause_detection()
                
                # Seek to frame
                with self.video_capture_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.video_position = target_frame
                self.total_frames = target_frame
                
                # Run detection on this frame (same as normal detection)
                self.run_detection_on_frame_simple(target_frame)
                
            except Exception as e:
                self.log_message(f"❌ Frame backward error: {e}")
    
    def frame_forward(self):
        """Move one frame forward and run detection"""
        if self.is_detecting and self.video_capture and not isinstance(self.video_path.get(), int):
            try:
                target_frame = min(self.video_total_frames - 1, self.total_frames + 1)
                
                # Pause detection
                if not self.is_paused:
                    self.pause_detection()
                
                # Seek to frame
                with self.video_capture_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.video_position = target_frame
                self.total_frames = target_frame
                
                # Run detection on this frame (same as normal detection)
                self.run_detection_on_frame_simple(target_frame)
                
            except Exception as e:
                self.log_message(f"❌ Frame forward error: {e}")
    
    def seek_to_frame(self, target_frame, auto_resume=True):
        """Seek to a specific frame number"""
        if self.is_detecting and self.video_capture and not isinstance(self.video_path.get(), int):
            try:
                # Temporarily pause detection to show frame preview
                was_paused = self.is_paused
                if not self.is_paused:
                    self.is_paused = True
                
                # Seek to target frame
                with self.video_capture_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    # Read and display the frame immediately
                    ret, frame = self.video_capture.read()
                self.video_position = target_frame
                self.total_frames = target_frame
                if ret:
                    # Process frame for display (without detection)
                    annotated_frame = frame.copy()
                    
                    # Add frame navigation indicator
                    cv2.putText(annotated_frame, f"FRAME: {target_frame}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Send frame to GUI immediately
                    if not self.frame_queue.full():
                        self.frame_queue.put(annotated_frame)
                
                # Update time labels and progress bar
                current_time = target_frame / self.video_fps
                progress_percent = (target_frame / self.video_total_frames) * 100
                
                self.current_time_label.config(text=f"{int(current_time//60):02d}:{int(current_time%60):02d}")
                self.frame_info_label.config(text=f"Frame: {target_frame} / {self.video_total_frames}")
                self.progress_bar.set(progress_percent)
                
                # Resume detection if it wasn't paused before and auto_resume is True
                if auto_resume and not was_paused:
                    self.is_paused = False
                    
            except Exception as e:
                self.log_message(f"❌ Seek to frame error: {e}")
    
    def on_seek_release(self, event):
        """Handle mouse release on progress bar to resume normal playback"""
        if self.is_detecting and not isinstance(self.video_path.get(), int):
            # Get the current progress bar value (frame position)
            current_progress = self.progress_bar.get()
            target_frame = int((float(current_progress) / 100.0) * self.video_total_frames)
            
            # Update the frame counters to match the seek position
            self.total_frames = target_frame
            self.video_position = target_frame
            
            # Set video capture to the correct frame
            if self.video_capture:
                with self.video_capture_lock:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            # Resume normal detection from the selected frame
            self.log_message(f"▶️ Resuming normal playback from frame {target_frame}")
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def stop_detection(self):
        """Stop video detection"""
        self.is_detecting = False
        self.is_paused = False
        self.main_control_button.config(text="🔴 Start Detection & Video")
        self.frame_back_button.config(state=tk.DISABLED)
        self.frame_forward_button.config(state=tk.DISABLED)
        self.stats_labels["Video Status"].config(text="⏸️ Stopped", foreground="orange")
        
        # Reset counters when stopping
        self.current_fps = 0
        self.stats_labels["Current FPS"].config(text="0.0")
        self.stats_labels["Avg Processing Time"].config(text="0.0ms")
        
        # Reset detection area tracking
        self.detection_area_center = None
        self.last_detection_center = None
        self.detection_area_size = None
        self.frames_since_last_detection = 0
        
        # Stop video capture
        if self.video_capture:
            with self.video_capture_lock:
                self.video_capture.release()
            self.video_capture = None
        
        # Clear frame queue to stop video display
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset video display to default message
        self.video_canvas.delete("all")
        self.video_canvas.create_text(320, 240, text="Video Stopped\nClick 'Start Detection' button or video area to resume", 
                                     fill='white', font=('Arial', 14), justify=tk.CENTER)
        
        # Reset progress bar
        self.progress_bar.config(state=tk.DISABLED)
        self.current_time_label.config(text="00:00")
        self.total_time_label.config(text="00:00")
        self.frame_info_label.config(text="Frame: 0 / 0")
        
        self.log_message("⏸️ Detection and video stopped!")
    
    def get_detection_area(self, frame, detections=None):
        """Get the detection area coordinates and size, centering around detected objects"""
        input_size_str = self.input_size.get()
        height, width = frame.shape[:2]
        
        # Handle Full Screen mode - return entire frame
        if input_size_str == "Full Screen":
            return (0, 0, width, height), (width, height)
        
        # Determine detection size based on input size setting
        if input_size_str == "Full Square":
            # Use the smaller dimension to ensure we don't exceed frame bounds
            detection_size = min(width, height)
            # Round to nearest multiple of 32 for YOLO compatibility
            detection_size = ((detection_size + 31) // 32) * 32
            # Ensure minimum size
            detection_size = max(detection_size, 416)
        else:
            # Convert string to integer
            try:
                detection_size = int(input_size_str)
            except (ValueError, TypeError):
                detection_size = 416  # Default fallback
        
        # CRITICAL FIX: Ensure detection size doesn't exceed frame dimensions
        # This prevents the detection area from being larger than the frame
        detection_size = min(detection_size, min(width, height))
        
        # Store the detection size
        self.detection_area_size = detection_size
        
        # Initialize center to frame center if not set
        if self.detection_area_center is None:
            self.detection_area_center = (width // 2, height // 2)
        
        # If we have detections, update the center to follow the object
        if detections and len(detections) > 0:
            # Use the first (highest confidence) detection
            detection = detections[0]
            bbox = detection['bbox']
            
            # Calculate center of the detected object
            obj_center_x = (bbox[0] + bbox[2]) // 2
            obj_center_y = (bbox[1] + bbox[3]) // 2
            
            self.last_detection_center = (obj_center_x, obj_center_y)
            self.detection_area_center = self.last_detection_center
            self.frames_since_last_detection = 0
        else:
            # No detections - keep using last known position
            self.frames_since_last_detection += 1
            
            # If we haven't seen an object for too long, reset to frame center
            if self.frames_since_last_detection > 30:  # 30 frames = ~1 second at 30fps
                self.detection_area_center = (width // 2, height // 2)
                self.last_detection_center = None
        
        # Calculate detection area bounds centered on the tracking point
        center_x, center_y = self.detection_area_center
        half_size = detection_size // 2
        
        # Calculate bounds, but don't let detection area get smaller when hitting edges
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, center_x + half_size)
        y2 = min(height, center_y + half_size)
        
        # Ensure we maintain the full detection size by adjusting position if needed
        if x2 - x1 < detection_size:
            if x1 == 0:
                x2 = min(width, detection_size)
            else:
                x1 = max(0, width - detection_size)
        
        if y2 - y1 < detection_size:
            if y1 == 0:
                y2 = min(height, detection_size)
            else:
                y1 = max(0, height - detection_size)
        
        return (x1, y1, x2, y2), detection_size
    
    def extract_detection_area(self, frame, detections=None):
        """Extract the detection area from frame and prepare for YOLO processing"""
        detection_coords, detection_size = self.get_detection_area(frame, detections)
        x1, y1, x2, y2 = detection_coords
        
        # Handle Full Screen mode
        if self.input_size.get() == "Full Screen":
            # For full screen, we return the entire frame
            detection_area = frame.copy()
            # No coordinate scaling needed for full screen
            return detection_area, detection_coords, detection_size
        
        # Extract the detection area (square for other modes)
        detection_area = frame[y1:y2, x1:x2]
        
        # Convert to grayscale if enabled (experimental - may cause errors)
        if self.use_grayscale.get():
            detection_area = cv2.cvtColor(detection_area, cv2.COLOR_BGR2GRAY)
        
        # No resizing needed - YOLO will handle any size adjustments internally
        # The cropped area is already the exact size we want (416x416, 512x512, etc.)
        
        return detection_area, detection_coords, detection_size
    
    def draw_detection_area_box(self, frame, detections=None):
        """Draw white box highlighting the detection area"""
        detection_coords, detection_size = self.get_detection_area(frame, detections)
        x1, y1, x2, y2 = detection_coords
        
        # For Full Screen mode, don't draw a box (entire frame is used)
        if self.input_size.get() == "Full Screen":
            # Just add a label indicating full screen mode
            label = "Detection Area: Full Screen (No Square Constraint)"
            font_scale = 0.7
            thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Position label at top of frame
            label_x = 10
            label_y = 30
            
            # Draw label background
            cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), 
                         (label_x + label_size[0] + 10, label_y + 5), (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (label_x + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            return frame
        
        # For square detection areas, draw the white rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
        # Add label showing detection area size
        if isinstance(detection_size, tuple):
            # Full screen mode (shouldn't reach here, but just in case)
            label = f"Detection Area: {detection_size[0]}x{detection_size[1]}"
        else:
            # Square mode
            label = f"Detection Area: {detection_size}x{detection_size}"
        
        font_scale = 0.7
        thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Position label above the box
        label_x = x1
        label_y = max(30, y1 - 10)
        
        # Draw label background
        cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 10, label_y + 5), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (label_x + 5, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def scale_boxes_to_frame(self, boxes, detection_coords, frame_shape):
        """Scale detection boxes from detection area coordinates to full frame coordinates"""
        if len(boxes) == 0:
            return boxes
        
        # For Full Screen mode, boxes are already in frame coordinates
        if self.input_size.get() == "Full Screen":
            # Just clip to frame bounds
            frame_height, frame_width = frame_shape[:2]
            clipped_boxes = []
            for box in boxes:
                frame_x1 = max(0, min(box[0], frame_width))
                frame_y1 = max(0, min(box[1], frame_height))
                frame_x2 = max(0, min(box[2], frame_width))
                frame_y2 = max(0, min(box[3], frame_height))
                clipped_boxes.append([frame_x1, frame_y1, frame_x2, frame_y2])
            return np.array(clipped_boxes)
        
        # For square detection areas, scale coordinates
        x1_det, y1_det, x2_det, y2_det = detection_coords
        detection_width = x2_det - x1_det
        detection_height = y2_det - y1_det
        
        scaled_boxes = []
        
        for box in boxes:
            # Box coordinates are relative to detection area (0 to detection_size)
            det_x1, det_y1, det_x2, det_y2 = box
            
            # Scale to detection area coordinates
            frame_x1 = x1_det + det_x1
            frame_y1 = y1_det + det_y1
            frame_x2 = x1_det + det_x2
            frame_y2 = y1_det + det_y2
            
            # Clip to frame bounds
            frame_height, frame_width = frame_shape[:2]
            frame_x1 = max(0, min(frame_x1, frame_width))
            frame_y1 = max(0, min(frame_y1, frame_height))
            frame_x2 = max(0, min(frame_x2, frame_width))
            frame_y2 = max(0, min(frame_y2, frame_height))
            
            scaled_boxes.append([frame_x1, frame_y1, frame_x2, frame_y2])
        
        return np.array(scaled_boxes)
    
    def track_objects(self, detections, frame_shape):
        """Simple object tracking using IoU"""
        if not self.enable_tracking.get():
            return detections
        
        current_objects = {}
        matched_tracks = set()
        
        for detection in detections:
            best_iou = 0
            best_track_id = None
            
            # Find best matching track
            for track_id, track_info in self.tracked_objects.items():
                if track_id in matched_tracks:
                    continue
                
                # Calculate IoU
                track_box = track_info['bbox']
                det_box = detection['bbox']
                
                # Calculate intersection
                x1 = max(track_box[0], det_box[0])
                y1 = max(track_box[1], det_box[1])
                x2 = min(track_box[2], det_box[2])
                y2 = min(track_box[3], det_box[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    union = track_area + det_area - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou and iou > 0.3:  # IoU threshold for tracking
                        best_iou = iou
                        best_track_id = track_id
            
            # Assign track ID
            if best_track_id is not None:
                detection['track_id'] = best_track_id
                current_objects[best_track_id] = detection
                matched_tracks.add(best_track_id)
            else:
                # New object
                self.track_id_counter += 1
                detection['track_id'] = self.track_id_counter
                current_objects[self.track_id_counter] = detection
        
        # Update tracked objects
        self.tracked_objects = current_objects
        
        return detections
    
    def detection_worker(self, video_source):
        """Optimized worker thread for video detection"""
        try:
            with self.video_capture_lock:
                self.video_capture = cv2.VideoCapture(video_source)
                if not self.video_capture.isOpened():
                    self.log_message("❌ Failed to open video source")
                    self.stop_detection()
                    return
            
            # Get video properties for progress bar
            if not isinstance(video_source, int):  # Not webcam
                self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 30  # Default FPS
                
                # Enable progress bar
                self.progress_bar.config(state=tk.NORMAL)
                self.seek_initialized = True  # Mark seek as initialized
                
                # Update total time label
                total_time = self.video_total_frames / self.video_fps
                self.total_time_label.config(text=self.format_time(total_time))
                self.frame_info_label.config(text=f"Frame: 0 / {self.video_total_frames}")
            
            fps_counter = 0
            fps_start_time = time.time()
            
            while self.is_detecting:
                # Check if paused
                while self.is_paused and self.is_detecting:
                    time.sleep(0.1)  # Small delay while paused
                
                # Ensure video capture position matches our frame counter
                with self.video_capture_lock:
                    if not isinstance(video_source, int) and self.video_total_frames > 0:
                        current_capture_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                        if abs(current_capture_frame - self.total_frames) > 1:
                            # Resynchronize if there's a mismatch
                            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames)
                    
                    ret, frame = self.video_capture.read()
                if not ret:
                    if isinstance(video_source, int):  # Webcam
                        continue
                    else:  # Video file ended
                        # Pause detection instead of stopping to allow seeking
                        self.log_message("📹 Video reached end - paused for seeking")
                        self.pause_detection()
                        # Keep the loop running to allow seeking
                        while self.is_detecting and self.is_paused:
                            time.sleep(0.1)
                        continue
                
                self.total_frames += 1
                self.video_position = self.total_frames
                fps_counter += 1
                
                # Update progress bar (only for video files, not webcam)
                if not isinstance(video_source, int) and self.video_total_frames > 0:
                    # Temporarily disable command to prevent seek callback
                    original_command = self.progress_bar['command']
                    self.progress_bar.config(command='')
                    
                    progress_percent = (self.total_frames / self.video_total_frames) * 100
                    self.progress_bar.set(progress_percent)
                    
                    # Re-enable command
                    self.progress_bar.config(command=original_command)
                    
                    # Update time labels
                    current_time = self.total_frames / self.video_fps
                    self.current_time_label.config(text=self.format_time(current_time))
                    self.frame_info_label.config(text=f"Frame: {self.total_frames} / {self.video_total_frames}")
                
                # FPS OPTIMIZATION: Two-pass detection for object tracking
                start_time = time.time()
                
                # First pass: Extract detection area based on previous detections or frame center
                detection_area, detection_coords, detection_size = self.extract_detection_area(frame, None)
                
                
                # Run detection with optimized parameters and GPU acceleration
                detection_kwargs = {
                    'conf': self.confidence.get(),
                    'iou': self.iou_threshold.get(),
                    'verbose': False,
                    'max_det': self.max_detections.get()
                }
                
                # Add GPU-specific optimizations
                try:
                    if self.use_gpu.get() and self.mixed_precision.get():
                        try:
                            import torch
                            with torch.amp.autocast('cuda'):
                                results = self.model(detection_area, **detection_kwargs)
                        except:
                            results = self.model(detection_area, **detection_kwargs)
                    else:
                        results = self.model(detection_area, **detection_kwargs)
                except Exception as e:
                    if "expected input" in str(e) and "channels" in str(e):
                        self.log_message("❌ Grayscale error: YOLO models expect 3-channel input. Disable grayscale mode.")
                        # Disable grayscale mode automatically
                        self.use_grayscale.set(False)
                        # Retry with color input
                        detection_area = frame[y1:y2, x1:x2]  # Re-extract without grayscale
                        results = self.model(detection_area, **detection_kwargs)
                    else:
                        raise e
                
                # Process results
                annotated_frame = frame.copy()
                detections = []
                
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    # Scale boxes back to original frame coordinates
                    boxes = self.scale_boxes_to_frame(boxes, detection_coords, frame.shape)
                    
                    # Filter for single object mode
                    if self.single_object_mode.get():
                        target_class = self.target_class.get()
                        filtered_indices = []
                        for i, class_id in enumerate(class_ids):
                            class_name = (list(self.model.names.values())[class_id] 
                                        if hasattr(self.model, 'names') else f"Class_{class_id}")
                            if class_name.lower() == target_class.lower():
                                filtered_indices.append(i)
                        
                        if filtered_indices:
                            boxes = boxes[filtered_indices]
                            confidences = confidences[filtered_indices]
                            class_ids = class_ids[filtered_indices]
                        else:
                            boxes = np.array([])
                            confidences = np.array([])
                            class_ids = np.array([])
                    
                    # Create detection objects
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = (list(self.model.names.values())[class_id] 
                                    if hasattr(self.model, 'names') else f"Class_{class_id}")
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })
                    
                    # RESPECT MAX DETECTIONS: Keep up to max_detections limit
                    if detections:
                        # Sort by confidence and keep up to max_detections
                        detections.sort(key=lambda x: x['confidence'], reverse=True)
                        max_det = self.max_detections.get()
                        detections = detections[:max_det]  # Keep up to max_detections
                    
                    # Apply object tracking
                    detections = self.track_objects(detections, frame.shape)
                
                # Update detection area center based on current detections for next frame
                self.get_detection_area(frame, detections)
                
                # Draw detection area box on frame
                annotated_frame = self.draw_detection_area_box(annotated_frame, detections)
                
                if detections:
                    # Draw detections
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        conf = detection['confidence']
                        class_name = detection['class']
                        track_id = detection.get('track_id', None)
                        
                        # Get color from model colors
                        if class_name.lower() in self.model_colors:
                            color = self.model_colors[class_name.lower()]
                        else:
                            color = self.colors[hash(class_name) % len(self.colors)]
                        
                        # Draw box
                        thickness = 3 if conf > 0.5 else 2
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label with tracking ID
                        label = f"{class_name.upper()}: {conf:.2f}"
                        if track_id:
                            label += f" ID:{track_id}"
                        
                        font_scale = 0.7 if conf > 0.5 else 0.6
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # Update detection count
                if detections:
                    self.detection_count += len(detections)
                    
                    # Log detection info with confidence values
                    detection_msg = f"Frame {self.total_frames}: {len(detections)} object - "
                    detection_msg += f"{processing_time:.1f}ms"
                    
                    # Add confidence details for each detection
                    conf_details = []
                    for det in detections:
                        conf_details.append(f"**{det['class']}({det['confidence']:.2f})**")
                    detection_msg += f" - [{', '.join(conf_details)}]"
                    
                    if self.single_object_mode.get():
                        detection_msg += f" (Single: {self.target_class.get()})"
                    
                    self.log_message(detection_msg[:120] + "..." if len(detection_msg) > 120 else detection_msg, 
                                   frame_number=self.total_frames, is_detection=True)
                
                # Calculate FPS only when detecting
                if self.is_detecting and fps_counter >= 10:
                    current_time = time.time()
                    self.current_fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                
                # Send frame to GUI - Clear queue if full to ensure fresh frames
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                    except queue.Empty:
                        pass
                self.frame_queue.put(annotated_frame)
                
                # Send stats to GUI only when detecting
                if self.is_detecting and not self.stats_queue.full():
                    avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                    
                    # Get current device name for stats
                    current_device = "CPU"
                    if self.use_gpu.get():
                        try:
                            import torch
                            if torch.cuda.is_available():
                                # Extract GPU ID from device selection
                                device_selection = self.gpu_device.get()
                                if "GPU" in device_selection:
                                    device_id = int(device_selection.split("GPU ")[1].split(")")[0])
                                else:
                                    device_id = 0
                                
                                current_device = torch.cuda.get_device_name(device_id)
                                if "4060" in current_device:
                                    current_device = "RTX 4060"
                                elif "4070" in current_device:
                                    current_device = "RTX 4070"
                                elif "4080" in current_device:
                                    current_device = "RTX 4080"
                                elif "4090" in current_device:
                                    current_device = "RTX 4090"
                        except:
                            current_device = "GPU"
                    
                    # Calculate actual detection area size for display
                    input_size_str = self.input_size.get()
                    if input_size_str == "Full Square":
                        # Get actual frame dimensions
                        if hasattr(self, 'video_capture') and self.video_capture:
                            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            min_dim = min(width, height)
                            actual_size = ((min_dim + 31) // 32) * 32
                            actual_size = max(actual_size, 416)
                            input_size_display = f"{actual_size}x{actual_size} (Full Square)"
                        else:
                            input_size_display = "Full Square (Auto)"
                    elif input_size_str == "Full Screen":
                        # Get actual frame dimensions
                        if hasattr(self, 'video_capture') and self.video_capture:
                            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            input_size_display = f"{width}x{height} (Full Screen)"
                        else:
                            input_size_display = "Full Screen (Auto)"
                    else:
                        input_size_display = f"{input_size_str}x{input_size_str}"
                    
                    stats = {
                        'total_frames': self.total_frames,
                        'detections': self.detection_count,
                        'fps': self.current_fps,
                        'avg_processing_time': avg_processing_time,
                        'input_size': input_size_display,
                        'tracking_objects': len(self.tracked_objects),
                        'gpu_memory': self.get_gpu_memory_usage(),
                        'device': current_device
                    }
                    self.stats_queue.put(stats)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
        
        except Exception as e:
            self.log_message(f"❌ Detection error: {e}")
        finally:
            if self.video_capture:
                self.video_capture.release()
            self.is_detecting = False
    
    def update_display(self):
        """Update GUI display with new frames and statistics"""
        # Update video frame
        try:
            # Process all available frames to get the latest one
            latest_frame = None
            while not self.frame_queue.empty():
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame is not None:
                # Resize frame to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    if latest_frame is not None and hasattr(latest_frame, 'shape') and len(latest_frame.shape) >= 2:
                        frame_height, frame_width = latest_frame.shape[:2]
                    else:
                        return
                    
                    # Calculate scaling to fit canvas while maintaining aspect ratio
                    scale_w = canvas_width / frame_width
                    scale_h = canvas_height / frame_height
                    scale = min(scale_w, scale_h)
                    
                    new_width = int(frame_width * scale)
                    new_height = int(frame_height * scale)
                    
                    resized_frame = cv2.resize(latest_frame, (new_width, new_height))
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    photo = ImageTk.PhotoImage(image=pil_image)
                    
                    # Update canvas
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                 image=photo, anchor=tk.CENTER)
                    self.video_canvas.image = photo
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display update error: {e}")
        
        # Update statistics
        try:
            if not self.stats_queue.empty():
                stats = self.stats_queue.get_nowait()
                self.stats_labels["Total Frames"].config(text=str(stats['total_frames']))
                self.stats_labels["Object Detections"].config(text=str(stats['detections']))
                self.stats_labels["Current FPS"].config(text=f"{stats['fps']:.1f}")
                self.stats_labels["Avg Processing Time"].config(text=f"{stats['avg_processing_time']:.1f}ms")
                self.stats_labels["Detection Area"].config(text=stats['input_size'])
                self.stats_labels["Tracking Objects"].config(text=str(stats['tracking_objects']))
                if 'gpu_memory' in stats and stats['gpu_memory']:
                    self.stats_labels["GPU Memory"].config(text=stats['gpu_memory'])
                if 'device' in stats:
                    self.stats_labels["Device"].config(text=stats['device'])
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(50, self.update_display)
    
    def get_gpu_memory_usage(self):
        """Get GPU memory usage if available"""
        try:
            import torch
            if torch.cuda.is_available() and self.use_gpu.get():
                # Extract GPU ID from device selection
                device_selection = self.gpu_device.get()
                if "GPU" in device_selection:
                    device_id = int(device_selection.split("GPU ")[1].split(")")[0])
                else:
                    device_id = 0
                
                memory_used = torch.cuda.memory_allocated(device_id) / 1024**3
                memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                memory_percent = (memory_used / memory_total) * 100
                
                # Add GPU name for display
                gpu_name = torch.cuda.get_device_name(device_id)
                if "4060" in gpu_name:
                    gpu_name = "RTX 4060"
                
                return f"{memory_used:.1f}/{memory_total:.1f}GB ({memory_percent:.0f}%)"
            else:
                return "CPU Mode"
        except (ImportError, AttributeError, RuntimeError):
            return "N/A"
    
    def save_current_frame(self):
        """Save current frame to file"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.queue[-1]
                timestamp = int(time.time())
                filename = f"optimized_detection_frame_{timestamp}.jpg"
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    initialfilename=filename,
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")]
                )
                
                if file_path:
                    cv2.imwrite(file_path, frame)
                    self.log_message(f"💾 Frame saved: {Path(file_path).name}")
                    messagebox.showinfo("Success", f"Frame saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save frame: {e}")
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_frames = 0
        self.detection_count = 0
        self.current_fps = 0
        self.processing_times = []
        self.tracked_objects = {}
        self.track_id_counter = 0
        
        self.stats_labels["Total Frames"].config(text="0")
        self.stats_labels["Object Detections"].config(text="0")
        self.stats_labels["Current FPS"].config(text="0.0")
        self.stats_labels["Avg Processing Time"].config(text="0.0ms")
        self.stats_labels["Tracking Objects"].config(text="0")
        
        # Clear detection log entries
        self.detection_log_entries = {}
        
        self.log_message("🔄 Statistics reset!")
    
    def on_log_click(self, event):
        """Handle clicks on detection log entries"""
        try:
            # Get click position
            click_index = self.log_text.index(f"@{event.x},{event.y}")
            
            # Check if click is on a detection entry
            for entry_id, entry_info in self.detection_log_entries.items():
                start_pos = entry_info['start']
                end_pos = entry_info['end']
                
                if self.log_text.compare(start_pos, "<=", click_index) and \
                   self.log_text.compare(click_index, "<=", end_pos):
                    
                    # Clicked on a detection entry
                    frame_number = entry_info['frame']
                    self.handle_detection_click(frame_number)
                    break
                    
        except Exception as e:
            print(f"Log click error: {e}")
    
    def on_log_hover(self, event):
        """Handle hover over detection log entries"""
        try:
            # Get hover position
            hover_index = self.log_text.index(f"@{event.x},{event.y}")
            
            # Reset all detection links to normal state
            self.log_text.tag_remove("detection_link_hover", "1.0", tk.END)
            self.log_text.tag_add("detection_link", "1.0", tk.END)
            
            # Check if hovering over a detection entry (frame number only)
            hovering_over_detection = False
            for entry_id, entry_info in self.detection_log_entries.items():
                start_pos = entry_info['start']
                end_pos = entry_info['end']
                
                if self.log_text.compare(start_pos, "<=", hover_index) and \
                   self.log_text.compare(hover_index, "<=", end_pos):
                    
                    # Hovering over a frame number - highlight it
                    self.log_text.tag_remove("detection_link", start_pos, end_pos)
                    self.log_text.tag_add("detection_link_hover", start_pos, end_pos)
                    hovering_over_detection = True
                    break
            
            # Change cursor based on hover state
            if hovering_over_detection:
                self.log_text.config(cursor="hand2")
            else:
                self.log_text.config(cursor="")
                    
        except Exception as e:
            pass  # Ignore hover errors
    
    def handle_detection_click(self, frame_number):
        """Handle click on detection log entry - pause, seek, and re-detect"""
        try:
            if not self.is_detecting or not self.video_capture:
                return
            
            # Pause detection
            was_paused = self.is_paused
            if not self.is_paused:
                self.pause_detection()
            
            # Run detection on this frame (same as normal detection)
            self.run_detection_on_frame_simple(frame_number)
            
            # Resume if it wasn't paused before
            if not was_paused:
                self.resume_detection()
                
        except Exception as e:
            self.log_message(f"❌ Detection click error: {e}")
    
    def on_video_canvas_click(self, event):
        """Handle click on video canvas to pause/unpause detection"""
        try:
            # Check if detection is running
            if not self.is_detecting:
                self.log_message("⚠️ Detection is not running. Click 'Start Detection' button first.")
                return
            
            # Toggle pause state
            if self.is_paused:
                self.log_message("▶️ Resuming detection via video canvas click")
                self.resume_detection()
            else:
                self.log_message("⏸️ Pausing detection via video canvas click")
                self.pause_detection()
                
        except Exception as e:
            self.log_message(f"❌ Video canvas click error: {e}")
    
    def run_detection_on_frame(self, frame_number):
        """Run detection on a specific frame"""
        try:
            if not self.video_capture or not self.model:
                return
            
            # Seek to frame
            with self.video_capture_lock:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.video_capture.read()
            
            if not ret:
                self.log_message(f"❌ Could not read frame {frame_number}")
                return
            
            # Run detection
            start_time = time.time()
            detection_area, detection_coords, detection_size = self.extract_detection_area(frame, None)
            
            # Run detection with current parameters
            detection_kwargs = {
                'conf': self.confidence.get(),
                'iou': self.iou_threshold.get(),
                'verbose': False,
                'max_det': self.max_detections.get()
            }
            
            results = self.model(detection_area, **detection_kwargs)
            
            # Process and display results
            annotated_frame = frame.copy()
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original frame coordinates
                boxes = self.scale_boxes_to_frame(boxes, detection_coords, frame.shape)
                
                # Filter for single object mode
                if self.single_object_mode.get():
                    target_class = self.target_class.get()
                    filtered_indices = []
                    for i, class_id in enumerate(class_ids):
                        class_name = (list(self.model.names.values())[class_id] 
                                    if hasattr(self.model, 'names') else f"Class_{class_id}")
                        if class_name.lower() == target_class.lower():
                            filtered_indices.append(i)
                    
                    if filtered_indices:
                        boxes = boxes[filtered_indices]
                        confidences = confidences[filtered_indices]
                        class_ids = class_ids[filtered_indices]
                    else:
                        boxes = np.array([])
                        confidences = np.array([])
                        class_ids = np.array([])
                
                # Create detection objects
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = (list(self.model.names.values())[class_id] 
                                if hasattr(self.model, 'names') else f"Class_{class_id}")
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                # RESPECT MAX DETECTIONS: Keep up to max_detections limit
                if detections:
                    detections.sort(key=lambda x: x['confidence'], reverse=True)
                    max_det = self.max_detections.get()
                    detections = detections[:max_det]  # Keep up to max_detections
            
            # Update detection area center based on current detections
            self.get_detection_area(frame, detections)
            
            # Draw detection area box on frame
            annotated_frame = self.draw_detection_area_box(annotated_frame, detections)
            
            if detections:
                # Draw detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    class_name = detection['class']
                    
                    # Ensure coordinates are integers and within frame bounds
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    height, width = annotated_frame.shape[:2]
                    x1 = max(0, min(x1, width-1))
                    y1 = max(0, min(y1, height-1))
                    x2 = max(0, min(x2, width-1))
                    y2 = max(0, min(y2, height-1))
                    
                    # Get color
                    if class_name.lower() in self.model_colors:
                        color = self.model_colors[class_name.lower()]
                    else:
                        color = self.colors[hash(class_name) % len(self.colors)]
                    
                    # Draw box with thicker lines for visibility
                    thickness = 4 if conf > 0.5 else 3
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label with better visibility
                    label = f"{class_name.upper()}: {conf:.2f}"
                    font_scale = 0.8 if conf > 0.5 else 0.7
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                    
                    # Draw label background
                    label_bg_x1 = x1
                    label_bg_y1 = max(0, y1 - label_size[1] - 10)
                    label_bg_x2 = x1 + label_size[0]
                    label_bg_y2 = y1
                    
                    cv2.rectangle(annotated_frame, (label_bg_x1, label_bg_y1), 
                                (label_bg_x2, label_bg_y2), color, -1)
                    cv2.rectangle(annotated_frame, (label_bg_x1, label_bg_y1), 
                                (label_bg_x2, label_bg_y2), (255, 255, 255), 2)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            # Add re-detection indicator
            if detections:
                cv2.putText(annotated_frame, f"RE-DETECTION: Frame {frame_number} - {len(detections)} object(s)", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, f"RE-DETECTION: Frame {frame_number} - No objects found", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Send frame to GUI
            if not self.frame_queue.full():
                self.frame_queue.put(annotated_frame)
            
            # Log re-detection results
            processing_time = (time.time() - start_time) * 1000
            if detections:
                conf_details = []
                for det in detections:
                    conf_details.append(f"{det['class']}({det['confidence']:.2f})")
                self.log_message(f"🔄 Re-detection on frame {frame_number}: {len(detections)} object - {processing_time:.1f}ms - [{', '.join(conf_details)}]")
                # Debug: Log bounding box coordinates
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    self.log_message(f"   📦 BBox: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            else:
                self.log_message(f"🔄 Re-detection on frame {frame_number}: No objects found - {processing_time:.1f}ms")
                
        except Exception as e:
                            self.log_message(f"❌ Re-detection error: {e}")
    
    def run_detection_on_frame_simple(self, frame_number):
        """Run detection on a specific frame (simplified - same as normal detection)"""
        try:
            if not self.video_capture or not self.model:
                return
            
            # Seek to frame
            with self.video_capture_lock:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.video_capture.read()
            
            if not ret:
                return
            
            # Run detection (same logic as normal detection loop)
            start_time = time.time()
            detection_area, detection_coords, detection_size = self.extract_detection_area(frame)
            
            # Run detection with current parameters
            detection_kwargs = {
                'conf': self.confidence.get(),
                'iou': self.iou_threshold.get(),
                'verbose': False,
                'max_det': self.max_detections.get()
            }
            
            results = self.model(detection_area, **detection_kwargs)
            
            # Process results (same as normal detection)
            annotated_frame = frame.copy()
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                # Scale boxes back to original frame coordinates
                boxes = self.scale_boxes_to_frame(boxes, detection_coords, frame.shape)
                
                # Filter for single object mode
                if self.single_object_mode.get():
                    target_class = self.target_class.get()
                    filtered_indices = []
                    for i, class_id in enumerate(class_ids):
                        class_name = (list(self.model.names.values())[class_id] 
                                    if hasattr(self.model, 'names') else f"Class_{class_id}")
                        if class_name.lower() == target_class.lower():
                            filtered_indices.append(i)
                    
                    if filtered_indices:
                        boxes = boxes[filtered_indices]
                        confidences = confidences[filtered_indices]
                        class_ids = class_ids[filtered_indices]
                    else:
                        boxes = np.array([])
                        confidences = np.array([])
                        class_ids = np.array([])
                
                # Create detection objects
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = (list(self.model.names.values())[class_id] 
                                if hasattr(self.model, 'names') else f"Class_{class_id}")
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
                
                # RESPECT MAX DETECTIONS: Keep up to max_detections limit
                if detections:
                    detections.sort(key=lambda x: x['confidence'], reverse=True)
                    max_det = self.max_detections.get()
                    detections = detections[:max_det]  # Keep up to max_detections
            
            # Update detection area center based on current detections
            self.get_detection_area(frame, detections)
            
            # Draw detection area box on frame
            annotated_frame = self.draw_detection_area_box(annotated_frame, detections)
            
            if detections:
                # Draw detections (same as normal detection)
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    class_name = detection['class']
                    
                    # Get color
                    if class_name.lower() in self.model_colors:
                        color = self.model_colors[class_name.lower()]
                    else:
                        color = self.colors[hash(class_name) % len(self.colors)]
                    
                    # Draw box
                    thickness = 3 if conf > 0.5 else 2
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label
                    label = f"{class_name.upper()}: {conf:.2f}"
                    font_scale = 0.7 if conf > 0.5 else 0.6
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Send frame to GUI
            if not self.frame_queue.full():
                self.frame_queue.put(annotated_frame)
            
            # Log detection (same format as normal detection)
            processing_time = (time.time() - start_time) * 1000
            if detections:
                self.detection_count += len(detections)
                
                # Log detection info with confidence values (same as normal detection)
                detection_msg = f"Frame {frame_number}: {len(detections)} object - "
                detection_msg += f"{processing_time:.1f}ms"
                
                # Add confidence details for each detection
                conf_details = []
                for det in detections:
                    conf_details.append(f"**{det['class']}({det['confidence']:.2f})**")
                detection_msg += f" - [{', '.join(conf_details)}]"
                
                if self.single_object_mode.get():
                    detection_msg += f" (Single: {self.target_class.get()})"
                
                self.log_message(detection_msg[:120] + "..." if len(detection_msg) > 120 else detection_msg, 
                               frame_number=frame_number, is_detection=True)
                
        except Exception as e:
            self.log_message(f"❌ Detection error: {e}")
    
    def log_message(self, message, frame_number=None, is_detection=False):
        """Add message to log with optional clickable detection entry and bold formatting"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            self.log_text.config(state=tk.NORMAL)
            
            if is_detection and frame_number is not None:
                # Create clickable detection entry with frame number as hyperlink
                timestamp_part = f"[{timestamp}] "
                frame_part = f"Frame {frame_number}"
                rest_part = message.replace(f"Frame {frame_number}", "", 1)  # Remove frame part from message
                
                # Insert timestamp
                self.log_text.insert(tk.END, timestamp_part)
                
                # Insert frame number as hyperlink
                frame_start = self.log_text.index(tk.END)
                self.log_text.insert(tk.END, frame_part)
                frame_end = self.log_text.index(tk.END)
                self.log_text.tag_add("detection_link", frame_start, frame_end)
                
                # Insert rest of message with bold formatting
                self._insert_formatted_text(rest_part + "\n")
                
                # Store frame number for click handling
                entry_id = f"{timestamp}_{frame_number}"
                self.detection_log_entries[entry_id] = {
                    'frame': frame_number,
                    'start': frame_start,
                    'end': frame_end
                }
            else:
                # Regular log entry (no formatting)
                self.log_text.insert(tk.END, log_entry)
            
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        except (tk.TclError, RuntimeError):
            pass
    
    def _insert_formatted_text(self, text):
        """Insert text with bold formatting for **text** markers"""
        import re
        
        # Find all **text** patterns
        bold_pattern = r'\*\*(.*?)\*\*'
        matches = list(re.finditer(bold_pattern, text))
        
        if not matches:
            # No bold markers found, insert as normal text
            self.log_text.insert(tk.END, text)
            return
        
        last_end = 0
        
        for match in matches:
            # Insert normal text before the bold part
            if match.start() > last_end:
                self.log_text.insert(tk.END, text[last_end:match.start()])
            
            # Insert bold text
            bold_start = self.log_text.index(tk.END)
            self.log_text.insert(tk.END, match.group(1))  # Insert without the ** markers
            bold_end = self.log_text.index(tk.END)
            self.log_text.tag_add("bold_text", bold_start, bold_end)
            
            last_end = match.end()
        
        # Insert any remaining normal text
        if last_end < len(text):
            self.log_text.insert(tk.END, text[last_end:])
    
    def export_log(self):
        """Export detection log to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                initialfilename=f"optimized_detection_log_{int(time.time())}.txt",
                filetypes=[("Text File", "*.txt"), ("All Files", "*.*")]
            )
            
            if file_path:
                log_content = self.log_text.get(1.0, tk.END)
                with open(file_path, 'w') as f:
                    f.write(log_content)
                
                self.log_message(f"📄 Log exported: {Path(file_path).name}")
                messagebox.showinfo("Success", f"Log exported to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export log: {e}")
    
    def find_default_files(self):
        """Find default model and video files in app directory"""
        # Look in the app's root directory (same level as the Python file)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for common model files
        model_patterns = ['*.pt', '*.pth', '*.onnx']
        for pattern in model_patterns:
            import glob
            models = glob.glob(os.path.join(app_dir, pattern))
            if models:
                # Use the first found model
                return models[0], None
        
        # Look for common video files
        video_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for pattern in video_patterns:
            import glob
            videos = glob.glob(os.path.join(app_dir, pattern))
            if videos:
                # Use the first found video
                return None, videos[0]
        
        return None, None

    def load_config(self):
        """Load configuration from config.json file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Load saved paths and settings
                if config.get('last_model_path') and os.path.exists(config['last_model_path']):
                    self.model_path.set(config['last_model_path'])
                
                if config.get('last_video_path') and os.path.exists(config['last_video_path']):
                    self.video_path.set(config['last_video_path'])
                
                # Load other settings
                self.confidence.set(config.get('confidence', 0.35))
                self.iou_threshold.set(config.get('iou_threshold', 0.4))
                self.input_size.set(config.get('input_size', 'Full Square'))
                self.max_detections.set(config.get('max_detections', 1))
                self.single_object_mode.set(config.get('single_object_mode', True))
                self.enable_tracking.set(config.get('enable_tracking', False))
                self.use_gpu.set(config.get('use_gpu', True))
                self.mixed_precision.set(config.get('mixed_precision', True))
                self.gpu_memory_fraction.set(config.get('gpu_memory_fraction', 0.8))
                
                print("✅ Configuration loaded from config.json")
            else:
                # No config file exists - try to find default files
                print("🔍 No config found, looking for default files...")
                default_model, default_video = self.find_default_files()
                
                if default_model:
                    self.model_path.set(default_model)
                    print(f"📁 Found default model: {os.path.basename(default_model)}")
                
                if default_video:
                    self.video_path.set(default_video)
                    print(f"📁 Found default video: {os.path.basename(default_video)}")
                
                # Set default settings
                self.confidence.set(0.35)
                self.iou_threshold.set(0.4)
                self.input_size.set('Full Square')
                self.max_detections.set(1)
                self.single_object_mode.set(True)
                self.enable_tracking.set(False)
                self.use_gpu.set(True)
                self.mixed_precision.set(True)
                self.gpu_memory_fraction.set(0.8)
                
                # Save the default config
                self.save_config()
                
        except Exception as e:
            print(f"⚠️ Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration to config.json file"""
        try:
            config = {
                'last_model_path': self.model_path.get(),
                'last_video_path': self.video_path.get(),
                'confidence': self.confidence.get(),
                'iou_threshold': self.iou_threshold.get(),
                'input_size': self.input_size.get(),
                'max_detections': self.max_detections.get(),
                'single_object_mode': self.single_object_mode.get(),
                'enable_tracking': self.enable_tracking.get(),
                'use_gpu': self.use_gpu.get(),
                'mixed_precision': self.mixed_precision.get(),
                'gpu_memory_fraction': self.gpu_memory_fraction.get()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            print("✅ Configuration saved to config.json")
        except Exception as e:
            print(f"⚠️ Could not save config: {e}")

    def on_closing(self):
        """Handle application closing"""
        if self.is_detecting:
            self.stop_detection()
        
        if self.video_capture:
            self.video_capture.release()
        
        # Save configuration before closing
        self.save_config()
        
        self.root.destroy()

def main():
    # Create the main window
    root = tk.Tk()
    
    # Set up the GUI
    app = YOLOTubeGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    print("🚀 Starting YOLO Tube...")
    root.mainloop()

if __name__ == "__main__":
    main()
