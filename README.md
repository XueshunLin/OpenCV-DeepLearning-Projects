# Computer Vision with OpenCV & Pytorch

This repository contains a collection of state-of-art projects focused on image processing and deep learning

## Projects

### 1. Image Compression with Discrete Wavelet Transform (DWT)
<a target="_blank" href="https://colab.research.google.com/github/Yagami11111/Pytorch-Computer-Vision/blob/main/image_compression/image_compression.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **Description:** Engineered 2D Haar wavelet transform algorithms to achieve efficient image compression, utilizing various filter banks for multi-level decomposition and reconstruction.
- **Technologies:** Python, NumPy, SciPy
- **Files:**
  - `image_compression/image_compression.ipynb`: Jupyter notebook with detailed implementation.
  - `image_compression/main.py`: Main script for the project.
  - `image_compression/test_image.png`: Sample image used for testing.

### 2. Image Rectification System
<a target="_blank" href="https://colab.research.google.com/github/Yagami11111/Pytorch-Computer-Vision/blob/main/image_rectification/image_rectification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **Description:** Implemented an image rectification system using OpenCV to correct perspective distortions and enhance image accuracy by applying denoising, edge detection, corner detection, thresholding, and more.
- **Technologies:** Python, Pillow, OpenCV
- **Files:**
  - `image_rectification/image_rectification.ipynb`: Jupyter notebook with detailed implementation.
  - `image_rectification/main.py`: Main script for the project.
  - `image_rectification/technical_report.pdf`: Technical report documenting the project.
  - `test_image`: Folder containing test images.

### 3. Deep Learning for Image Classification
 <a target="_blank" href="https://colab.research.google.com/github/Yagami11111/Pytorch-Computer-Vision/blob/main/image_classification/image_classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **Description:** Designed and optimized a ResNet-based CNN to classify images into 15 categories. Achieved high accuracy through extensive hyperparameter tuning with Optuna, data augmentation, early stopping, and OneCycleLR scheduling.
- **Technologies:** Python, PyTorch, Torchvision, Optuna, scikit-learn
- **Files:**
  - `image_classification/image_classification.ipynb`: Jupyter notebook with detailed implementation.
  - `image_classification/main.py`: Main script for the project.
  - `image_classification/ResNet_model.pt`: Trained ResNet model.
  - `image_classification/technical_report.pdf`: Technical report documenting the project.
  - `image_classification/train_image.zip`: Dataset used for training.