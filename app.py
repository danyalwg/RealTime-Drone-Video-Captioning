import sys
import cv2
import torch
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QTabWidget, QLineEdit, QComboBox, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from transformers import BlipProcessor, BlipForConditionalGeneration
import qdarkstyle

# Import Flask for the terminal log web server
from flask import Flask, render_template_string

# Global variables for terminal logs and thread safety
terminal_logs = []
terminal_logs_lock = threading.Lock()

# Set up the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    with terminal_logs_lock:
        logs = terminal_logs.copy()
    logs_str = "\n".join(logs)
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="refresh" content="5">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Terminal Logs</title>
      <style>
        body {
          background-color: #121212;
          color: #e0e0e0;
          font-family: Consolas, "Courier New", monospace;
          padding: 20px;
        }
        pre {
          white-space: pre-wrap;
          word-wrap: break-word;
          font-size: 14px;
        }
        h1 {
          font-size: 24px;
        }
      </style>
    </head>
    <body>
      <h1>Terminal Logs</h1>
      <pre>{{ logs }}</pre>
    </body>
    </html>
    '''
    return render_template_string(html, logs=logs_str)

def run_flask():
    # Run the Flask app. Use host='0.0.0.0' to allow access from other devices if needed.
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


class CaptionThread(QThread):
    caption_generated = pyqtSignal(str)

    def __init__(self, get_latest_frame, processor, model, parent=None):
        super().__init__(parent)
        self.get_latest_frame = get_latest_frame
        self.processor = processor
        self.model = model
        self.running = True

    def run(self):
        while self.running:
            frame = self.get_latest_frame()
            if frame is None:
                self.msleep(50)
                continue

            try:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("Error converting frame:", e)
                continue

            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.processor(image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = self.model.generate(**inputs)
            caption = self.processor.decode(output[0], skip_special_tokens=True)

            self.caption_generated.emit(caption)
            self.msleep(100)  # Adjust this to control inference frequency

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Drone Video Captioning")
        self.resize(1000, 800)

        # Video source variables
        self.cap = None
        self.source_type = None  # "file", "rtsp", or "usb"
        self.fileName = ""
        self.rtspUrl = ""
        self.usb_index = None  # For USB camera

        # Frame and metric variables
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0  # For video FPS calculation
        self.start_time = None

        # Inference (captioning) metrics
        self.inference_count = 0
        self.inference_start_time = time.time()

        self.init_ui()

        # Timer to update video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Load the BLIP model and processor
        self.processor, self.model = self.load_model()

        # Start the caption generation thread
        self.caption_thread = CaptionThread(self.get_latest_frame, self.processor, self.model)
        self.caption_thread.caption_generated.connect(self.update_caption)
        self.caption_thread.start()

    def init_ui(self):
        # Top-level tab widget with "Main" and "Terminal" tabs.
        self.top_tab_widget = QTabWidget()

        # ---- Main Tab: Contains video display, controls, metrics, and input selection ----
        main_tab = QWidget()
        main_layout = QVBoxLayout()

        # Sub-tab widget for input selection: File, RTSP, USB Camera.
        self.input_tab_widget = QTabWidget()

        # File Input Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        self.file_select_button = QPushButton("Select Video File")
        self.file_select_button.clicked.connect(self.open_file)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_select_button)
        file_layout.addWidget(self.file_label)
        file_tab.setLayout(file_layout)

        # RTSP Input Tab
        rtsp_tab = QWidget()
        rtsp_layout = QVBoxLayout()
        self.rtsp_lineedit = QLineEdit()
        self.rtsp_lineedit.setPlaceholderText("Enter RTSP URL here")
        self.rtsp_connect_button = QPushButton("Connect Stream")
        self.rtsp_connect_button.clicked.connect(self.connect_rtsp)
        self.rtsp_status_label = QLabel("Not connected")
        rtsp_layout.addWidget(self.rtsp_lineedit)
        rtsp_layout.addWidget(self.rtsp_connect_button)
        rtsp_layout.addWidget(self.rtsp_status_label)
        rtsp_tab.setLayout(rtsp_layout)

        # USB Camera Input Tab
        usb_tab = QWidget()
        usb_layout = QVBoxLayout()
        self.usb_combo = QComboBox()
        self.populate_usb_cameras()  # Populate drop-down with available webcams
        self.usb_connect_button = QPushButton("Connect Camera")
        self.usb_connect_button.clicked.connect(self.connect_usb)
        self.usb_status_label = QLabel("Not connected")
        usb_layout.addWidget(QLabel("Select USB Camera:"))
        usb_layout.addWidget(self.usb_combo)
        usb_layout.addWidget(self.usb_connect_button)
        usb_layout.addWidget(self.usb_status_label)
        usb_tab.setLayout(usb_layout)

        self.input_tab_widget.addTab(file_tab, "File")
        self.input_tab_widget.addTab(rtsp_tab, "RTSP")
        self.input_tab_widget.addTab(usb_tab, "USB Camera")
        main_layout.addWidget(self.input_tab_widget)

        # Control Buttons and Status
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        self.status_label = QLabel("Status: Idle")
        control_layout.addWidget(self.status_label)
        main_layout.addLayout(control_layout)

        # Video Display Area
        self.video_label = QLabel("Video will be displayed here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)

        # Caption Display Area
        self.caption_label = QLabel("Caption will appear here")
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setStyleSheet("font-size: 18pt;")
        main_layout.addWidget(self.caption_label)

        # Metrics Display (only FPS and Inference FPS)
        self.metrics_label = QLabel("FPS: 0.00 | Inference FPS: 0.00")
        main_layout.addWidget(self.metrics_label)

        main_tab.setLayout(main_layout)
        self.top_tab_widget.addTab(main_tab, "Main")

        # ---- Terminal Tab for Log Output ----
        terminal_tab = QWidget()
        terminal_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        terminal_layout.addWidget(self.log_text)
        terminal_tab.setLayout(terminal_layout)
        self.top_tab_widget.addTab(terminal_tab, "Terminal")

        self.setCentralWidget(self.top_tab_widget)

    def populate_usb_cameras(self):
        """
        Scans for available USB cameras by trying to open video capture for indices 0 to 10.
        Adds the available indices to the combo box.
        """
        self.usb_combo.clear()
        available_cameras = []
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap is not None and cap.isOpened():
                available_cameras.append(index)
                cap.release()
        if available_cameras:
            for cam in available_cameras:
                self.usb_combo.addItem(f"Camera {cam}", cam)
        else:
            self.usb_combo.addItem("No cameras found", -1)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        with terminal_logs_lock:
            terminal_logs.append(log_message)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_name:
            self.fileName = file_name
            self.source_type = "file"
            self.file_label.setText(f"Selected file: {file_name}")
            self.log("Video file selected.")

    def connect_rtsp(self):
        url = self.rtsp_lineedit.text().strip()
        if url:
            self.rtspUrl = url
            self.source_type = "rtsp"
            self.rtsp_status_label.setText("RTSP URL set.")
            self.log("RTSP URL set: " + url)
        else:
            self.rtsp_status_label.setText("Please enter a valid RTSP URL.")
            self.log("Invalid RTSP URL entered.")

    def connect_usb(self):
        cam_index = self.usb_combo.currentData()
        if cam_index is None or cam_index == -1:
            self.usb_status_label.setText("No valid camera selected.")
            self.log("No valid USB camera selected.")
        else:
            self.usb_index = cam_index
            self.source_type = "usb"
            self.usb_status_label.setText(f"Selected: Camera {cam_index}")
            self.log(f"USB camera selected: Camera {cam_index}")

    def start_video(self):
        if self.source_type == "file" and self.fileName:
            self.cap = cv2.VideoCapture(self.fileName)
            self.log("Starting video from file.")
        elif self.source_type == "rtsp" and self.rtspUrl:
            self.cap = cv2.VideoCapture(self.rtspUrl)
            self.log("Starting video from RTSP stream.")
        elif self.source_type == "usb" and self.usb_index is not None:
            self.cap = cv2.VideoCapture(self.usb_index)
            self.log(f"Starting video from USB camera (Camera {self.usb_index}).")
        else:
            self.status_label.setText("Status: No video source selected.")
            self.log("No video source selected.")
            return

        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText("Status: Failed to open video source.")
            self.log("Failed to open video source.")
            return

        self.frame_count = 0
        self.start_time = time.time()
        # Reset inference metrics
        self.inference_count = 0
        self.inference_start_time = time.time()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        interval = int(1000 / fps)
        self.timer.start(interval)
        self.status_label.setText("Status: Video started.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log(f"Video started with FPS: {fps}")

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_label.setText("Status: Video stopped.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("Video stopped.")

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            self.status_label.setText("Status: End of video or stream lost.")
            self.log("End of video or stream lost.")
            return

        with self.frame_lock:
            self.latest_frame = frame.copy()
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        video_fps = self.frame_count / elapsed if elapsed > 0 else 0
        inference_elapsed = time.time() - self.inference_start_time
        inference_fps = (self.inference_count / inference_elapsed) if inference_elapsed > 0 else 0
        self.metrics_label.setText(f"FPS: {video_fps:.2f} | Inference FPS: {inference_fps:.2f}")

        # Convert frame to displayable format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def update_caption(self, caption):
        self.caption_label.setText(caption)
        self.inference_count += 1
        self.log("Caption: " + caption)

    def load_model(self):
        self.log("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        self.log("Fine Tuned BLIP model loaded.")
        return processor, model

    def closeEvent(self, event):
        self.timer.stop()
        if self.caption_thread:
            self.caption_thread.stop()
            self.caption_thread.wait()
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    # Start the Flask server in a separate daemon thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Start the PyQt5 application
    app_qt = QApplication(sys.argv)
    app_qt.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app_qt.exec_())
