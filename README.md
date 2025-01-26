# Face Mask Detection 😷🔍

## Overview 🌐
This project implements a real-time face mask detection system using deep learning techniques. It leverages TensorFlow's Keras API and OpenCV to classify and detect face masks in live video streams.

## Features ✨
- 🤖 Convolutional Neural Network (CNN) for mask detection
- 📹 Real-time video stream analysis
- �顔 Face detection using Haar Cascades
- 🚦 Color-coded mask/no-mask indicators

## Requirements 🛠
- Python 3.x
- TensorFlow
- OpenCV
- NumPy

## Installation 💻
```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
```

## Usage 🚀
1. Train the model:
```python
python train_model.py
```

2. Run real-time detection:
```python
python detect_masks.py
```

## Model Architecture 🧠
- Input Layer: 150x150x3 image
- 3 Convolutional Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers with ReLU and Sigmoid activations

## Contributions 🤝
Contributions, issues, and feature requests are welcome!

## License 📄
MIT License
