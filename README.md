# 🔥 Fire Detection App - Multi-Source Camera Support

A comprehensive fire and smoke detection application using YOLO models with support for multiple camera sources including local cameras, WiFi cameras, mobile IP cameras, and Android ScreenStream.

## ✨ Features

- 🎯 **YOLO Fire/Smoke Detection** - Real-time detection with custom model support
- 📱 **Multiple Camera Sources**:
  - Local USB/Built-in cameras
  - WiFi/IP cameras (RTSP)
  - Mobile IP cameras (DroidCam, IP Webcam, iVCam, EpocCam)
  - Android ScreenStream
- 🚀 **GPU Acceleration** - CUDA, Apple Silicon (MPS), and CPU support
- 📹 **Recording & Snapshots** - Save detection sessions and capture frames
- 🔍 **Network Scanning** - Auto-detect cameras on your network
- 🔊 **Alert System** - Audio alerts when fire/smoke is detected
- 🖥️ **User-Friendly GUI** - PyQt6-based interface

## 🛠️ Requirements

### System Requirements
- **Windows**: Windows 10/11
- **macOS**: macOS 10.15+ (Apple Silicon supported)
- **Linux**: Ubuntu 18.04+ or equivalent
- **Python**: 3.8 - 3.11
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support

### Hardware Requirements
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB free space
- **Camera**: Any supported camera source
- **Network**: For IP cameras and ScreenStream

## 📦 Installation

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd fire-detection-app
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

#### For NVIDIA GPU (CUDA) - Recommended
```bash
# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install ultralytics opencv-python PyQt6 playsound numpy
```

#### For CPU Only or Mac
```bash
# Install PyTorch (CPU/MPS for Apple Silicon)
pip install torch torchvision torchaudio

# Install other dependencies
pip install ultralytics opencv-python PyQt6 playsound numpy
```

#### Using Requirements File
Create a `requirements.txt` file:

**For CUDA:**
```text
--index-url https://download.pytorch.org/whl/cu121

torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
ultralytics~=8.3.126
opencv-python~=4.11.0.86
PyQt6~=6.9.0
playsound
numpy
```

**For CPU/Mac:**
```text
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
ultralytics~=8.3.126
opencv-python~=4.11.0.86
PyQt6~=6.9.0
playsound
numpy
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### 1. Download a YOLO Model
You need a YOLO model trained for fire/smoke detection. You can:
- Use a pre-trained model (search for "fire detection YOLO model")
- Train your own using Ultralytics YOLO
- Use a general YOLO model for testing

### 2. Run the Application
```bash
python main.py
```

### 3. Load Your Model
1. Click **"Load Model"** button
2. Select your `.pt` model file
3. Wait for model loading confirmation

### 4. Select Camera Source
Choose from:
- **Local Camera**: Built-in or USB cameras
- **WiFi Camera**: Enter RTSP URL
- **Mobile IP Camera**: Use network scan or manual IP
- **ScreenStream**: Android screen streaming

### 5. Start Detection
1. Configure your camera settings
2. Click **"Start"** to begin detection
3. Enable recording if desired

## 📱 Camera Setup Guides

### Local Camera
- Simply select from available cameras in the dropdown
- Grant camera permissions if prompted

### WiFi/IP Camera
```
Format: rtsp://username:password@ip_address:port/path
Example: rtsp://admin:password123@192.168.1.100:554/stream1
```

### Mobile IP Cameras

#### DroidCam (Android)
1. Install DroidCam on your Android device
2. Start DroidCam and note the IP address
3. Use network scan or enter IP manually
4. Default port: 4747

#### IP Webcam (Android)
1. Install IP Webcam app
2. Start server and note the IP
3. Default port: 8080
4. Use format: `http://IP:8080/video`

#### iVCam (iOS)
1. Install iVCam on iOS device and PC client
2. Connect devices to same network
3. App should auto-detect

### ScreenStream (Android)
1. Install ScreenStream app on Android
2. Start streaming and note IP:port
3. Use auto-detect or enter manually: `http://192.168.1.100:8080`
4. Set PIN if required

## 🎛️ Configuration

### Detection Settings
- **Confidence Threshold**: Adjust detection sensitivity (1-100%)
- **Recording**: Enable to save detection sessions
- **Fullscreen**: Toggle fullscreen display

### Performance Optimization
- **GPU**: Automatically detected (CUDA/MPS/CPU)
- **Model Size**: Use smaller models (YOLOv8n) for real-time performance
- **Resolution**: Lower camera resolution for better performance

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"
```

#### CUDA Not Available
- Verify NVIDIA drivers are installed
- Check CUDA toolkit version compatibility
- Reinstall PyTorch with CUDA support

#### Network Camera Connection Failed
- Verify IP address and port
- Check network connectivity
- Ensure camera is on same network
- Test camera URL in VLC or browser

#### ScreenStream Connection Issues
- Ensure Android device and PC are on same WiFi
- Check firewall settings
- Try both MJPEG and static JPEG methods
- Verify ScreenStream app is running

#### Model Loading Errors
- Ensure model file is valid YOLO format (.pt)
- Check model compatibility with Ultralytics version
- Verify sufficient RAM/VRAM

### Performance Issues
- **Reduce camera resolution**
- **Lower confidence threshold**
- **Use smaller YOLO model (YOLOv8n vs YOLOv8l)**
- **Close other applications**
- **Enable GPU acceleration**

### Debug Mode
Add debug prints by modifying the camera source change function to see what's happening:
```python
print(f"Camera source changed to: {self.camera_source}")
```

## 📊 Model Training (Optional)

To train your own fire detection model:

1. **Collect Data**: Gather fire/smoke images and videos
2. **Annotate**: Use tools like LabelImg or Roboflow
3. **Train**: Use Ultralytics YOLO
```bash
yolo train data=your_dataset.yaml model=yolov8n.pt epochs=100
```
4. **Export**: Save trained model as `.pt` file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for YOLO implementation
- **PyQt6** for GUI framework
- **OpenCV** for computer vision
- **PyTorch** for deep learning

## 📞 Support

For issues and questions:
1. Check troubleshooting section
2. Search existing issues
3. Create new issue with:
   - System information
   - Error messages
   - Steps to reproduce

## 🔄 Updates

Check for updates regularly:
```bash
pip install --upgrade ultralytics
```

---

**🔥 Happy Fire Detection! Stay Safe! 🔥**
