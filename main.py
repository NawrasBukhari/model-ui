import os
import socket
import sys
import threading
import time
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any

import cv2
import torch
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QFileDialog, QSlider, QSizePolicy, QLineEdit,
    QGroupBox, QFormLayout, QRadioButton, QButtonGroup, QMessageBox,
    QProgressDialog
)
from ultralytics import YOLO


class CameraSource(Enum):
    LOCAL = auto()
    WIFI = auto()
    MOBILE = auto()


class MobileCameraApp:
    def __init__(self, name: str, port: int, path: str):
        self.name = name
        self.port = port
        self.path = path


MOBILE_CAMERA_APPS = [
    MobileCameraApp("IP Webcam (Android)", 1443, "/h264_pcm.sdp"),
    MobileCameraApp("DroidCam (Android)", 4747, "/video"),
    MobileCameraApp("iVCam (iOS)", 1234, "/"),
    MobileCameraApp("EpocCam (iOS)", 5555, "/stream")
]


class DetectionState(Enum):
    IDLE = auto()
    SCANNING = auto()
    RUNNING = auto()
    ERROR = auto()


class NetworkScanner(QThread):
    """Thread for scanning the local network for devices"""
    update_progress = pyqtSignal(int)
    scan_complete = pyqtSignal(list)

    def __init__(self, ip_range: Optional[List[str]] = None):
        super().__init__()
        self.ip_range = ip_range or self._get_local_network()
        self.found_devices: List[Dict[str, str]] = []
        self.is_running = True

    @staticmethod
    def _get_local_network() -> List[str]:
        """Get the local IP range for scanning"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]

            ip_parts = local_ip.split('.')
            ip_prefix = '.'.join(ip_parts[0:3])
            return [f"{ip_prefix}.{i}" for i in range(1, 255)]
        except Exception:
            return [f"192.168.1.{i}" for i in range(1, 255)]

    def run(self):
        """Scan the network for devices with mobile camera app ports open"""
        for i, ip in enumerate(self.ip_range):
            if not self.is_running:
                break

            progress = int((i / len(self.ip_range)) * 100)
            self.update_progress.emit(progress)  # type: ignore

            for application in MOBILE_CAMERA_APPS:
                if self._check_port(ip, application.port):
                    device_name = f"{application.name} ({ip})"
                    url = f"http://{ip}:{application.port}{application.path}"
                    self.found_devices.append({"name": device_name, "url": url})
                    break

        self.scan_complete.emit(self.found_devices)  # type: ignore

    @staticmethod
    def _check_port(ip: str, port: int) -> bool:
        """Check if a specific port is open on a given IP"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                result = sock.connect_ex((ip, port))
                return result == 0
        except Exception:
            return False

    def stop(self):
        """Stop the scanning process"""
        self.is_running = False


class VideoHandler:
    """Handles all video-related operations"""

    def __init__(self):
        self.capture = None
        self.video_writer = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.record_enabled = False

    def open_camera(self, source: Any) -> bool:
        """Open a video capture source"""
        self.release()
        self.capture = cv2.VideoCapture(source)
        return self.capture.isOpened()

    def start_recording(self, filename: str) -> bool:
        """Start video recording"""
        if not self.capture or not self.capture.isOpened():
            return False

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        self.record_enabled = True
        return self.video_writer.isOpened()

    def write_frame(self, frame):
        """Write a frame to the video file"""
        if self.record_enabled and self.video_writer:
            self.video_writer.write(frame)

    def release(self):
        """Release all video resources"""
        if self.capture:
            self.capture.release()
            self.capture = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.record_enabled = False

    def get_frame(self) -> Optional[Any]:
        """Get the current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def set_frame(self, frame):
        """Set the current frame"""
        with self.frame_lock:
            self.current_frame = frame.copy()

    def read_frame(self) -> tuple:
        """Read a frame from the capture"""
        if not self.capture or not self.capture.isOpened():
            return False, None
        return self.capture.read()


class DetectionThread(QThread):
    """Thread for running YOLO detection"""
    detection_complete = pyqtSignal(object, bool)

    def __init__(self, model: Any, video_handler: VideoHandler, conf_threshold: float, device: str):
        super().__init__()
        self.model = model
        self.video_handler = video_handler
        self.conf_threshold = conf_threshold
        self.device = device
        self.running = False

    def run(self):
        """Main detection loop"""
        self.running = True
        while self.running:
            frame = self.video_handler.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    device=self.device
                )

                detection = False
                for r in results:
                    boxes = r.boxes
                    names = r.names
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls.item())
                            label = names[cls_id].lower()
                            if "fire" in label or "smoke" in label:
                                detection = True
                                break

                self.detection_complete.emit(results, detection)  # type: ignore
            except Exception as e:
                print(f"Detection error: {e}")

            time.sleep(0.01)

    def stop(self):
        """Stop the detection thread"""
        self.running = False


class FireDetectionApp(QWidget):
    """Main application class for fire/smoke detection"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Fire/Smoke Detection")

        self.device = self._get_device()
        self.model = None
        self.model_path = None
        self.video_handler = VideoHandler()
        self.detection_thread = None
        self.network_scanner = None
        self.progress_dialog = None
        self.mobile_devices = []

        self.state = DetectionState.IDLE
        self.conf_threshold = 0.1
        self.alert_interval = 5
        self.last_alert_time = 0
        self.alert_triggered = False
        self.latest_results = None
        self.camera_source = CameraSource.LOCAL

        self.init_ui()
        self.update_ui_state()

    @staticmethod
    def _get_device() -> str:
        """Determine the best available device (CUDA GPU or CPU)"""
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            return f"{device} ({device_name})"
        elif torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"

    def init_ui(self):
        """Initialize the user interface"""

        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.device_info_label = QLabel(f"Using device: {self.device}")
        self.device_info_label.setStyleSheet("color: green; font-weight: bold;")

        self.camera_source_group = QGroupBox("Camera Source")
        self._init_camera_source_ui()

        self.camera_settings_box = QGroupBox("Camera Settings")
        self._init_camera_settings_ui()

        self._init_control_buttons()

        self.status_label = QLabel("Status: Ready")
        self.detection_counts_label = QLabel("Detections: None")

        layout = QVBoxLayout()
        layout.addWidget(self.device_info_label)
        layout.addWidget(self.video_label)

        source_settings_layout = QHBoxLayout()
        source_settings_layout.addWidget(self.camera_source_group)
        source_settings_layout.addWidget(self.camera_settings_box)
        layout.addLayout(source_settings_layout)

        layout.addLayout(self.control_buttons_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.detection_counts_label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # type: ignore

    def _init_camera_source_ui(self):
        """Initialize camera source selection UI"""
        layout = QVBoxLayout()

        self.local_camera_radio = QRadioButton("Local Camera")
        self.wifi_camera_radio = QRadioButton("WiFi Camera")
        self.mobile_camera_radio = QRadioButton("Mobile IP Camera")
        self.local_camera_radio.setChecked(True)

        button_group = QButtonGroup()
        button_group.addButton(self.local_camera_radio)
        button_group.addButton(self.wifi_camera_radio)
        button_group.addButton(self.mobile_camera_radio)
        button_group.buttonClicked.connect(self._on_camera_source_changed)  # type: ignore

        layout.addWidget(self.local_camera_radio)
        layout.addWidget(self.wifi_camera_radio)
        layout.addWidget(self.mobile_camera_radio)
        self.camera_source_group.setLayout(layout)

    def _init_camera_settings_ui(self):
        """Initialize camera settings UI"""
        layout = QFormLayout()

        self.camera_selector = QComboBox()
        self._populate_camera_list()
        layout.addRow("Local Camera:", self.camera_selector)

        self.wifi_url_input = QLineEdit()
        self.wifi_url_input.setPlaceholderText("rtsp://username:password@ip_address:port/path")
        self.test_connection_button = QPushButton("Test Connection")
        self.test_connection_button.clicked.connect(self.test_connection)  # type: ignore
        wifi_layout = QHBoxLayout()
        wifi_layout.addWidget(self.wifi_url_input)
        wifi_layout.addWidget(self.test_connection_button)
        layout.addRow("WiFi Camera URL:", wifi_layout)

        self.mobile_camera_selector = QComboBox()
        self.mobile_camera_selector.setPlaceholderText("Select a mobile camera...")
        self.mobile_scan_button = QPushButton("Scan Network")
        self.mobile_scan_button.clicked.connect(self.scan_for_mobile_cameras)  # type: ignore
        mobile_scan_layout = QHBoxLayout()
        mobile_scan_layout.addWidget(self.mobile_camera_selector)
        mobile_scan_layout.addWidget(self.mobile_scan_button)
        layout.addRow("Mobile Camera:", mobile_scan_layout)

        self.mobile_manual_ip = QLineEdit()
        self.mobile_manual_ip.setPlaceholderText("Or enter IP address manually (e.g. 192.168.1.100)")
        layout.addRow("Manual IP:", self.mobile_manual_ip)

        self.camera_settings_box.setLayout(layout)

    def _init_control_buttons(self):
        """Initialize control buttons"""
        self.control_buttons_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_detection)  # type: ignore
        self.snapshot_button = QPushButton("Save Snapshot")
        self.snapshot_button.clicked.connect(self.save_snapshot)  # type: ignore
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)  # type: ignore

        self.record_checkbox = QCheckBox("Enable Recording")
        self.full_width_checkbox = QCheckBox("Fullscreen Display")
        self.full_width_checkbox.stateChanged.connect(self.toggle_fullscreen)  # type: ignore

        self.conf_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_threshold_slider.setMinimum(1)
        self.conf_threshold_slider.setMaximum(100)
        self.conf_threshold_slider.setValue(int(self.conf_threshold * 100))
        self.conf_threshold_slider.valueChanged.connect(self.update_threshold)  # type: ignore
        self.conf_threshold_label = QLabel(f"{self.conf_threshold_slider.value()}%")

        self.control_buttons_layout.addWidget(self.start_button)
        self.control_buttons_layout.addWidget(self.snapshot_button)
        self.control_buttons_layout.addWidget(self.load_model_button)
        self.control_buttons_layout.addWidget(self.record_checkbox)
        self.control_buttons_layout.addWidget(self.full_width_checkbox)
        self.control_buttons_layout.addWidget(QLabel("Conf Threshold"))
        self.control_buttons_layout.addWidget(self.conf_threshold_slider)
        self.control_buttons_layout.addWidget(self.conf_threshold_label)

    def _on_camera_source_changed(self, button):
        """Handle camera source selection change"""
        if button == self.local_camera_radio:
            self.camera_source = CameraSource.LOCAL
        elif button == self.wifi_camera_radio:
            self.camera_source = CameraSource.WIFI
        else:
            self.camera_source = CameraSource.MOBILE
        self.update_ui_state()

    def update_ui_state(self):
        """Update UI elements based on current state"""

        self.camera_selector.setEnabled(self.camera_source == CameraSource.LOCAL)
        self.wifi_url_input.setEnabled(self.camera_source == CameraSource.WIFI)
        self.test_connection_button.setEnabled(self.camera_source == CameraSource.WIFI)
        self.mobile_camera_selector.setEnabled(self.camera_source == CameraSource.MOBILE)
        self.mobile_scan_button.setEnabled(self.camera_source == CameraSource.MOBILE)
        self.mobile_manual_ip.setEnabled(self.camera_source == CameraSource.MOBILE)

        self.start_button.setText("Start" if self.state != DetectionState.RUNNING else "Stop")

    def _populate_camera_list(self):
        """Populate the local camera selector with available cameras"""
        self.camera_selector.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_selector.addItem(f"Camera {i}")
            cap.release()

    def update_threshold(self):
        """Update the confidence threshold from slider value"""
        self.conf_threshold = self.conf_threshold_slider.value() / 100.0
        self.conf_threshold_label.setText(f"{self.conf_threshold_slider.value()}%")

    def load_model(self):
        """Load a YOLO model from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", "YOLO Model Files (*.pt)"
        )
        if not file_path:
            return

        try:
            self.model_path = file_path
            device = self.device.split()[0].lower()
            self.model = YOLO(self.model_path)
            self.model.to(device)
            self.status_label.setText(f"✅ Loaded model: {file_path.split('/')[-1]} on {self.device}")
        except Exception as e:
            self.status_label.setText(f"❌ Failed to load model: {str(e)}")
            QMessageBox.critical(
                self, "Model Loading Error",
                f"Error loading model: {str(e)}\n\nDetails: {sys.exc_info()[2]}"
            )

    def save_snapshot(self):
        """Save the current frame as an image"""
        frame = self.video_handler.get_frame()
        if frame is not None:
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            self.status_label.setText(f"📸 Snapshot saved: {filename}")

    def toggle_detection(self):
        """Start or stop the detection process"""
        if self.state == DetectionState.RUNNING:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        """Start the detection process"""
        if not self.model:
            self.status_label.setText("⚠️ Please load a model first.")
            return

        if self.camera_source == CameraSource.LOCAL:
            if not self.camera_selector.currentText():
                self.status_label.setText("⚠️ No local camera available")
                return
            cam_index = int(self.camera_selector.currentText().split()[-1])
            source_desc = f"Camera {cam_index}"
            success = self.video_handler.open_camera(cam_index)
        elif self.camera_source == CameraSource.WIFI:
            url = self.wifi_url_input.text().strip()
            if not url:
                self.status_label.setText("⚠️ Please enter a WiFi camera URL")
                return
            source_desc = f"WiFi camera at {url}"
            success = self.video_handler.open_camera(url)
        else:
            url = self._get_mobile_camera_url()
            if not url:
                self.status_label.setText("⚠️ Please select a mobile camera or enter an IP address")
                return
            source_desc = f"Mobile camera at {url}"
            success = self.video_handler.open_camera(url)

        if not success:
            self.status_label.setText(f"⚠️ Failed to open {source_desc}")
            return

        if self.record_checkbox.isChecked():
            filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            self.video_handler.start_recording(filename)

        device = self.device.split()[0].lower()
        self.detection_thread = DetectionThread(
            self.model, self.video_handler, self.conf_threshold, device
        )
        self.detection_thread.detection_complete.connect(self._on_detection_complete)  # type: ignore
        self.detection_thread.start()

        self.timer.start(30)
        self.state = DetectionState.RUNNING
        self.status_label.setText(f"✅ Detection started on {source_desc}")
        self.update_ui_state()

    def stop_detection(self):
        """Stop the detection process"""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.detection_thread = None

        self.video_handler.release()
        self.timer.stop()
        self.state = DetectionState.IDLE
        self.status_label.setText("🛑 Detection stopped")
        self.update_ui_state()

    def _on_detection_complete(self, results, detection):
        """Handle detection results from the thread"""
        self.latest_results = results
        self.alert_triggered = detection

        if results and results[0].boxes is not None:
            counts = len(results[0].boxes)
            self.detection_counts_label.setText(f"Detections: {counts}")
        else:
            self.detection_counts_label.setText("Detections: None")

    def update_frame(self):
        """Update the video frame display"""
        ret, frame = self.video_handler.read_frame()
        if not ret:
            if self.state == DetectionState.RUNNING:
                self.status_label.setText("⚠️ Camera disconnected")
                self.stop_detection()
            return

        if self.camera_source == CameraSource.LOCAL:
            frame = cv2.flip(frame, 1)

        self.video_handler.set_frame(frame)

        annotated_frame = self._annotate_frame(frame)

        self.video_handler.write_frame(annotated_frame)

        self._display_frame(annotated_frame)

        self._play_alert_sound()

    def _annotate_frame(self, frame):
        """Annotate the frame with detection results and timestamp"""

        if self.latest_results:
            frame = self.latest_results[0].plot()

        if self.alert_triggered:
            cv2.putText(frame, "FIRE/SMOKE DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def _display_frame(self, frame):
        """Display the frame in the UI"""
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def _play_alert_sound(self):
        """Play alert sound using playsound package (works on all platforms)"""
        if (self.alert_triggered and
                (time.time() - self.last_alert_time > self.alert_interval)):
            try:
                from playsound import playsound

                sound_file = os.path.join(os.path.dirname(__file__), "alert.wav")

                threading.Thread(
                    target=playsound,
                    args=(sound_file, True),
                    daemon=True
                ).start()

                self.last_alert_time = time.time()
            except Exception as e:
                print(f"Error playing alert sound: {e}")
                print("\a")

    def test_connection(self):
        """Test the connection to a WiFi camera"""
        url = self.wifi_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a valid camera URL")
            return

        self.status_label.setText("Testing connection...")
        QApplication.processEvents()

        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self._display_frame(frame)
                self.status_label.setText("✅ Camera connection successful!")
                QMessageBox.information(
                    self, "Connection Successful",
                    "Successfully connected to the WiFi camera."
                )
            else:
                self.status_label.setText("❌ Failed to connect to camera")
                QMessageBox.warning(
                    self, "Connection Failed",
                    "Could not connect to the WiFi camera. Please check the URL."
                )
        cap.release()

    def scan_for_mobile_cameras(self):
        """Scan the local network for mobile IP cameras"""
        if self.network_scanner and self.network_scanner.isRunning():
            return

        self.progress_dialog = QProgressDialog(
            "Scanning network for mobile cameras...",
            "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Network Scan")
        self.progress_dialog.setModal(True)

        self.network_scanner = NetworkScanner()
        self.network_scanner.update_progress.connect(self.progress_dialog.setValue)  # type: ignore
        self.network_scanner.scan_complete.connect(self._on_scan_complete)  # type: ignore
        self.progress_dialog.canceled.connect(self.network_scanner.stop)  # type: ignore

        self.network_scanner.start()
        self.progress_dialog.show()
        self.status_label.setText("🔍 Scanning network for mobile cameras...")
        self.state = DetectionState.SCANNING

    def _on_scan_complete(self, devices):
        """Handle network scan completion"""
        self.progress_dialog.hide()
        self.mobile_devices = devices
        self.state = DetectionState.IDLE

        self.mobile_camera_selector.clear()
        if not devices:
            self.status_label.setText("⚠️ No mobile cameras found on the network")
            return

        for device in devices:
            self.mobile_camera_selector.addItem(device["name"], device["url"])

        self.status_label.setText(f"✅ Found {len(devices)} mobile camera(s) on the network")

    def _get_mobile_camera_url(self) -> Optional[str]:
        """Get the URL for the selected mobile camera"""

        if self.mobile_camera_selector.currentIndex() >= 0:
            return self.mobile_camera_selector.currentData()

        manual_ip = self.mobile_manual_ip.text().strip()
        if manual_ip:
            for app in MOBILE_CAMERA_APPS:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(0.5)
                        result = sock.connect_ex((manual_ip, app.port))
                        if result == 0:
                            return f"http://{manual_ip}:{app.port}{app.path}"
                except Exception:
                    continue

            return f"http://{manual_ip}:4747/video"

        return None

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.full_width_checkbox.isChecked():
            self.showFullScreen()
        else:
            self.showNormal()

    def closeEvent(self, event):
        """Handle application close event"""
        if self.network_scanner and self.network_scanner.isRunning():
            self.network_scanner.stop()

        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        win = FireDetectionApp()
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {e}")
        import traceback

        traceback.print_exc()
