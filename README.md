# Brain Tumor Segmentation and Classification (Graduation Project)

## Overview

This project focuses on developing a AI Model for brain tumor segmentation and classification using Glioma MRI images. The project leverages deep learning techniques to accurately segment and classify brain tumors, utilizing a combination of popular Python libraries including PyTorch, Keras, OpenCV, and more.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To install the necessary dependencies, you can use the following commands:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn seaborn tqdm h5py nibabel opencv-python scipy keras
pip install timm einops ml_collections wget tensorboardX SimpleITK medpy
pip install -U datasets trl accelerate peft bitsandbytes transformers trl huggingface_hub
git clone "https://huggingface.co/Unknown6197/BEFUnet_Brats2020"
```

## Dataset

The dataset used in this project is the BraTS2020 dataset, which contains MRI images of brain tumors. The dataset is divided into training and test sets.

**Training Data:** Located at `/kaggle/input/brats2020-training-data`.

## My Contributing

Role: Classification Part
Responsibilities:
Led the classification model development.
Preprocessed and analyzed the dataset.
Designed and implemented the classification algorithm.
Optimized model performance and validated results.
Integrated the classification component into the overall system.
