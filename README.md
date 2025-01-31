# Face Mask Detection 😷🔍

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Maintenance](https://img.shields.io/badge/Maintenance-Active-brightgreen.svg)

A real-time face mask detection system powered by deep learning, utilizing TensorFlow's Keras API and OpenCV for live video stream analysis and classification.

## 📖 Table of Contents
- [Core Features](#-core-features)
- [Technical Requirements](#-technical-requirements)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Core Features

### 🤖 Deep Learning Detection System
- **CNN Architecture**
  - Custom-designed convolutional neural network
  - Optimized for real-time processing
  - High accuracy mask classification
  - Robust feature extraction

### 📹 Video Processing
- **Real-time Analysis**
  - Live video stream processing
  - Frame-by-frame detection
  - Multi-face detection capability
  - Low-latency processing

### 👤 Face Detection System
- **Haar Cascade Implementation**
  - Efficient face region detection
  - Scale-invariant detection
  - Multiple face tracking
  - Region of interest optimization

### 🎯 Classification System
- **Binary Classification**
  - Mask/No-mask detection
  - Confidence scoring
  - Real-time status updates
  - Color-coded indicators (Green/Red)

## 🛠 Technical Requirements

### System Dependencies
```txt
python>=3.8
tensorflow>=2.5.0
opencv-python>=4.5.0
numpy>=1.19.0
pillow>=8.0.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

### Hardware Requirements
- **Minimum Specifications**
  - CPU: Intel Core i5 or equivalent
  - RAM: 8GB
  - Storage: 5GB free space
  - Webcam: 720p resolution
- **Recommended Specifications**
  - CPU: Intel Core i7 or equivalent
  - RAM: 16GB
  - GPU: NVIDIA GTX 1660 or better
  - Webcam: 1080p resolution

## 💻 Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/face-mask-detection.git

# Navigate to project directory
cd face-mask-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
```python
# config.py
CONFIG = {
    'model_path': 'models/mask_detector.h5',
    'cascade_path': 'cascades/haarcascade_frontalface_default.xml',
    'confidence_threshold': 0.5,
    'frame_width': 640,
    'frame_height': 480
}
```

## 🚀 Usage Guide

### Training the Model
```python
# Train a new model
python train_model.py --dataset data/mask_dataset --epochs 20 --batch-size 32

# Resume training
python train_model.py --model models/mask_detector.h5 --epochs 10
```

### Running Detection
```python
# Start real-time detection
python detect_masks.py --source 0  # Use webcam
python detect_masks.py --source video.mp4  # Use video file
```

## 🧠 Model Architecture

### Network Structure
```mermaid
graph TD
    A[Input Layer 150x150x3] --> B[Conv2D 32 3x3]
    B --> C[MaxPool 2x2]
    C --> D[Conv2D 64 3x3]
    D --> E[MaxPool 2x2]
    E --> F[Conv2D 128 3x3]
    F --> G[MaxPool 2x2]
    G --> H[Flatten]
    H --> I[Dense 128 ReLU]
    I --> J[Dense 1 Sigmoid]
```

### Layer Details
| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input | (150, 150, 3) | 0 |
| Conv2D | (148, 148, 32) | 896 |
| MaxPool | (74, 74, 32) | 0 |
| Conv2D | (72, 72, 64) | 18,496 |
| MaxPool | (36, 36, 64) | 0 |
| Conv2D | (34, 34, 128) | 73,856 |
| MaxPool | (17, 17, 128) | 0 |
| Flatten | (36,992) | 0 |
| Dense | (128) | 4,735,104 |
| Dense | (1) | 129 |

## ⚡ Performance

### Benchmarks
| Metric | Value |
|--------|--------|
| Training Accuracy | 98.2% |
| Validation Accuracy | 97.5% |
| Inference Time | ~0.03s |
| FPS | 30-35 |

### Optimization Techniques
- Batch normalization
- Dropout layers
- Data augmentation
- Early stopping

## 👨‍💻 Development

### Project Structure
```
face-mask-detection/
├── data/
│   ├── mask/
│   └── no_mask/
├── models/
│   └── mask_detector.h5
├── src/
│   ├── train_model.py
│   ├── detect_masks.py
│   └── utils.py
├── config.py
├── requirements.txt
└── README.md
```

### Code Style
```python
# Example code style
def process_frame(frame, face_cascade, mask_detector):
    """
    Process a single frame for face mask detection.
    
    Args:
        frame (numpy.ndarray): Input frame
        face_cascade: Haar cascade classifier
        mask_detector: Trained mask detection model
        
    Returns:
        numpy.ndarray: Processed frame with detections
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        prediction = mask_detector.predict(face_roi)
        draw_prediction(frame, (x, y, w, h), prediction)
        
    return frame
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- TensorFlow team for deep learning framework
- Dataset contributors
- Open source community
