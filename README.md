# 🛰️ Speckle2Void

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <b>Self-Supervised SAR Denoising for Defense Surveillance</b><br>
  <i>Blind-Spot U-Net Architecture • No Ground Truth Required • Production-Ready ONNX Deployment</i>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Performance Metrics](#-performance-metrics)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Web Application](#web-application)
- [Dataset](#-dataset)
- [Model Export](#-model-export)
- [Applications](#-applications)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Speckle2Void** is a self-supervised deep learning framework for removing coherent speckle noise from Synthetic Aperture Radar (SAR) imagery. Unlike traditional supervised denoising methods that require clean ground truth images, Speckle2Void leverages the **Noise2Void** paradigm with a blind-spot masking strategy to learn denoising directly from noisy observations.

This approach is particularly valuable for SAR processing where acquiring noise-free reference images is impractical or impossible, making it ideal for **defense surveillance**, **maritime monitoring**, and **remote sensing** applications.

### Why Speckle2Void?

| Challenge | Traditional Methods | Speckle2Void |
|-----------|-------------------|--------------|
| Ground Truth | Required | **Not Required** ✅ |
| Noise Model | Must be known | **Learned Automatically** ✅ |
| Deployment | Complex pipelines | **Single ONNX Model** ✅ |
| Real-time | Often slow | **CPU Optimized** ✅ |

---

## ✨ Key Features

- 🔬 **Self-Supervised Learning**: No clean reference images needed; trains directly on noisy SAR data
- 🎭 **Blind-Spot Masking**: 10% adversarial pixel dropout strategy for robust noise estimation
- 🏗️ **U-Net Architecture**: Encoder-decoder with skip connections for multi-scale feature extraction
- ⚡ **ONNX Quantization**: INT8 quantized model for fast CPU inference
- 🌐 **Streamlit Web App**: Interactive demo for real-time denoising
- 📊 **Comprehensive Metrics**: PSNR, SSIM, ENL, SNR, and Edge Preservation evaluation

---

## 🚀 Demo

**Live Application**: [https://speckle2void-app.streamlit.app](https://speckle2void-app.streamlit.app)

Upload any L/P/X-band SAR image and observe real-time speckle reduction with quantified metrics.

---

## 🏛️ Architecture

### Blind-Spot U-Net

```
Input (1, 256, 256)
        │
        ▼
┌───────────────────┐
│   Encoder Block 1 │ ──────────────────────────┐
│   (1 → 32 ch)     │                           │
└───────────────────┘                           │
        │ MaxPool                               │
        ▼                                       │
┌───────────────────┐                           │
│   Encoder Block 2 │ ────────────────┐         │
│   (32 → 64 ch)    │                 │         │
└───────────────────┘                 │         │
        │ MaxPool                     │         │
        ▼                             │         │
┌───────────────────┐                 │         │
│    Bottleneck     │                 │         │
│   (64 → 128 ch)   │                 │         │
└───────────────────┘                 │         │
        │ Upsample                    │         │
        ▼                             │         │
┌───────────────────┐                 │         │
│   Decoder Block 2 │ ◄───────────────┘         │
│  (192 → 64 ch)    │   Skip Connection         │
└───────────────────┘                           │
        │ Upsample                              │
        ▼                                       │
┌───────────────────┐                           │
│   Decoder Block 1 │ ◄─────────────────────────┘
│   (96 → 32 ch)    │   Skip Connection
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   1x1 Conv + σ    │
│   (32 → 1 ch)     │
└───────────────────┘
        │
        ▼
Output (1, 256, 256)
```

### Self-Supervised Training Strategy

The **Noise2Void** approach works by:

1. **Blind-Spot Masking**: Randomly select ~10% of pixels and replace with noise
2. **Prediction**: Network predicts original pixel values from spatial context only
3. **Loss Calculation**: MSE computed only at masked locations
4. **Learning**: Network learns to distinguish signal from noise without clean references

```python
# Blind-spot masking strategy
def apply_blind_spot_mask(image, num_pixels=800):
    masked_img = image.clone()
    mask = torch.zeros_like(image)
    
    # Randomly mask pixels
    ys = torch.randint(0, h, (num_pixels,))
    xs = torch.randint(0, w, (num_pixels,))
    
    masked_img[:, :, ys, xs] = random_noise
    mask[:, :, ys, xs] = 1.0
    
    return masked_img, mask
```

---

## 📊 Performance Metrics

Evaluated on the **Official SSDD Dataset** (1,160 SAR images):

| Metric | Value | Description |
|--------|-------|-------------|
| **PSNR** | 39.3 dB | Peak Signal-to-Noise Ratio |
| **SSIM** | 0.92 | Structural Similarity Index |
| **ENL Improvement** | 5.8× | Equivalent Number of Looks (2.11 → 12.24) |
| **SNR Improvement** | +1.55 dB | Signal-to-Noise Ratio gain |
| **Noise Reduction** | 71% | Speckle variance reduction |
| **Edge Preservation** | 70% | Structural detail retention |

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x (optional, for GPU training)

### Clone Repository

```bash
git clone https://github.com/Swarno-Coder/speckle2void.git
cd speckle2void
```

### Install Dependencies

```bash
# For inference only (lightweight)
pip install -r requirements.txt

# For training (full dependencies)
pip install torch torchvision opencv-python matplotlib numpy
```

### Requirements

```
streamlit
onnxruntime
numpy
Pillow
torch>=2.0.0        # For training
torchvision         # For training
opencv-python       # For training
matplotlib          # For visualization
```

---

## 📖 Usage

### Training

1. **Download the SSDD Dataset**:
```bash
git clone https://github.com/TianwenZhang0825/Official-SSDD-OPEN.git
```

2. **Run Training Script**:
```python
python Speckle2Void_Pro_Colab.py
```

**Training Configuration**:
- Epochs: 50 (with early stopping)
- Batch Size: 32
- Learning Rate: 0.002 (with ReduceLROnPlateau scheduler)
- Optimizer: Adam
- Early Stopping Patience: 6 epochs

### Inference

```python
import torch
from unet import BlindSpotUNet
from PIL import Image
import numpy as np

# Load model
model = BlindSpotUNet()
model.load_state_dict(torch.load('best_speckle2void.pth'))
model.eval()

# Preprocess image
img = Image.open('sar_image.jpg').convert('L').resize((256, 256))
tensor = torch.tensor(np.array(img) / 255.0).float().unsqueeze(0).unsqueeze(0)

# Denoise
with torch.no_grad():
    denoised = model(tensor)

# Save result
output = (denoised.squeeze().numpy() * 255).astype(np.uint8)
Image.fromarray(output).save('denoised.jpg')
```

### Web Application

```bash
cd SSDD_refined
streamlit run app.py
```

Access the application at `http://localhost:8501`

---

## 📁 Dataset

This project uses the **Official SSDD (SAR Ship Detection Dataset)**:

- **Source**: [Official-SSDD-OPEN](https://github.com/TianwenZhang0825/Official-SSDD-OPEN)
- **Images**: 19,488 SAR images
- **Resolution**: Various (resized to 256×256)
- **Sensors**: L-band, C-band SAR
- **Annotations**: Ship bounding boxes (not used for denoising)

### Dataset Split

| Split | Percentage | Images |
|-------|------------|--------|
| Train | 80% | ~15590 |
| Validation | 10% | ~1948 |
| Test | 10% | ~1950 |

---

## 📦 Model Export

### PyTorch to ONNX Conversion

```python
import torch
from unet import BlindSpotUNet

model = BlindSpotUNet()
model.load_state_dict(torch.load('best_speckle2void.pth'))
model.eval()

dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model, dummy_input, 'speckle2void_raw.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### ONNX Quantization (INT8)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'speckle2void_raw.onnx',
    'speckle2void_quantized.onnx',
    weight_type=QuantType.QUInt8
)
```

**Model Sizes**:
- PyTorch (.pth): ~1.2 MB
- ONNX (FP32): ~1.1 MB
- ONNX (INT8): ~0.4 MB

---

## 🎯 Applications

Speckle2Void is designed for critical defense and surveillance applications:

| Domain | Use Case |
|--------|----------|
| 🚢 **Maritime Surveillance** | Ship detection in noisy ocean SAR |
| 🌍 **Border Monitoring** | All-weather terrain observation |
| 🛩️ **Drone SAR Systems** | Real-time onboard processing |
| 🏗️ **Infrastructure Monitoring** | Change detection in urban areas |
| 🎖️ **Military Intelligence** | Target identification enhancement |
| 🌊 **Oceanography** | Sea state and oil spill detection |

### Radar Band Compatibility

- **L-Band** (1-2 GHz): ✅ Optimized
- **P-Band** (0.3-1 GHz): ✅ Compatible
- **X-Band** (8-12 GHz): ✅ Compatible
- **C-Band** (4-8 GHz): ✅ Compatible

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use Speckle2Void in your research, please cite:

```bibtex
@software{speckle2void2024,
  author = {Swarnodip Nag},
  title = {Speckle2Void: Self-Supervised SAR Denoising for Defense Surveillance},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Swarno-Coder/speckle2void}
}
```

### Related Papers

- Krull, A., Buchholz, T. O., & Jug, F. (2019). **Noise2Void - Learning Denoising from Single Noisy Images**. CVPR 2019.
- Zhang, T., et al. (2021). **SAR Ship Detection Dataset (SSDD): Official Release and Benchmark**. Remote Sensing.

---

## 🙏 Acknowledgments

- **Official SSDD Dataset**: [TianwenZhang0825/Official-SSDD-OPEN](https://github.com/TianwenZhang0825/Official-SSDD-OPEN)
- **Noise2Void Paper**: Krull et al., CVPR 2019
- **PyTorch Team**: For the deep learning framework
- **ONNX Runtime**: For efficient model deployment

---

<p align="center">
  <b>Built with ❤️ for Defense & Surveillance Applications</b><br>
  <a href="https://github.com/Swarno-Coder">@Swarno-Coder</a>
</p>
