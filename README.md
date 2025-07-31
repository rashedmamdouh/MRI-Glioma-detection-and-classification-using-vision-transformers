# MRI-Glioma Detection & Classification with Vision Transformers (Academic Collaborations)

Advanced brain tumor segmentation and glioma-classification using Vision Transformers (ViTs) and hybrid architectures, implemented with PyTorch and Keras.

## 👁️‍🗨️ Overview

This repo targets robust **segmentation** and **glioma classification** from multi-sequence MRI scans using state-of-the-art Vision Transformer models and Resnet CNN pipelines.  

---

## ✨ Key Features 

- **Tumor segmentation** using transformer-based models (BEFUnet, Swin-UNet, etc.) 
- **Glioma classification** based on segmented regions: tumor grading or subtyping  
- Multi-sequence MRI input support: T1, T1‑CE, T2, FLAIR
- End-to-end pipeline: preprocessing — segmentation — classification — visualization  
- **Flask API** for inference served via REST endpoints

---

## 📂 Repository Contents

```

/
├── README.md
├── flask/                # REST API and backend
├── BEFUnet\_Brats2020/    # Pretrained segmentation models
├── NoteBook.ipynb        # Experiment notebook
├── DocumentationBook.pdf # Project report

````

- **NoteBook.ipynb** — Data loading, model testing, inference examples  
- **BEFUnet_Brats2020/** — pretrained segmentation weights for BraTS2020 trained BEFUnet  
- **flask/** — Flask application supporting REST inferencing

---

## 🛠️ Installation & Setup 

```bash
git clone https://github.com/rashedmamdouh/MRI-Glioma-detection-and-classification-using-vision-transformers.git
cd MRI-Glioma-…‍
pip install torch torchvision numpy pandas matplotlib scikit-learn seaborn tqdm h5py nibabel opencv-python scipy keras timm einops datasets tensorboardX simpleitk medpy fastapi flask transformers
````

**Data Preparation**
Use the **BraTS 2020** dataset (requires registration). Place multi-modal MRI scans in folders as expected by the notebook or flask scripts. ([GitHub][1], [GitHub][2])

---

## 🚀 Usage Examples

### 1. Segmentation & Classification (Notebook)

Open and run `NoteBook.ipynb` to:

* Preprocess MRI sequences
* Segment tumors with BEFUnet or Swin-UNet
* Classify tumor region using custom classification module
* Visualize segmentation masks and classification labels

### 2. Flask API for Serving Models

Start the REST API:

```bash
cd flask
python app.py
```

Call the `/detect` or `/classify` endpoint with a multi-sequence MRI input:

```bash
curl -X POST http://localhost:5000/classify -F "image=@/path/to/mri.nii"
```

Supports real-time inference and result visualization. ([GitHub][3], [GitHub][1])

---

## 📊 Metrics & Results

* Segmentation: BEFUnet achieves \~**0.80 mIoU** on BraTS2020 ([GitHub][2])
* Classification: Reported accuracies around **90–95%** for glioma sub-typing with fine‑tuned ViT models in literature ([أرشيف أرآيف][4], [GitHub][5])

Include confusion matrices, Dice scores, and classification accuracy plots in your documentation.

---

## 🧩 Customization & Extensions

* Add Hodgkin or novel transformer models like Swin UNETR, TransBTS or ResMT ([أرشيف أرآيف][6], [sciencedirect.com][7])
* Fine-tune the ViT classification module on glioma subtypes (e.g., LGG vs HGG)
* Extend to real-time UI using React or Next.js + Flask backend
* Replace backend with FastAPI for async deployment

---

## 📃 Project Structure 

| Component               | Description                                  |
| ----------------------- | -------------------------------------------- |
| `flask/`                | API services for segmentation/classification |
| `BEFUnet_Brats2020/`    | Pretrained segmentation model files          |
| `NoteBook.ipynb`        | Demo and experiment scripts                  |
| `DocumentationBook.pdf` | Full report of methodology and outcomes      |

---

[1]: https://github.com/rashedmamdouh/MRI-Glioma-detection-and-classification-using-vision-transformers?utm_source=chatgpt.com "rashedmamdouh/MRI-Glioma-detection-and-classification-using-vision ..."
[2]: https://github.com/OptimusAI01/Brain-MRI-Segmentation?utm_source=chatgpt.com "GitHub - OptimusAI01/Brain-MRI-Segmentation"
[3]: https://github.com/ousidus/glioma-detection-visual-transformers?utm_source=chatgpt.com "ousidus/glioma-detection-visual-transformers - GitHub"
[4]: https://arxiv.org/abs/2502.20715?utm_source=chatgpt.com "Glioma Classification using Multi-sequence MRI and Novel Wavelets-based ..."
[5]: https://github.com/saraaburomoh/Fine-tuning-VIT-on-MRI-images?utm_source=chatgpt.com "Fine-tuning Vision Transformer (ViT) on MRI Images - GitHub"
[6]: https://arxiv.org/abs/2103.04430?utm_source=chatgpt.com "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer"
[7]: https://www.sciencedirect.com/science/article/pii/S0045790624006724?utm_source=chatgpt.com "ResMT: A hybrid CNN-transformer framework for glioma grading with 3D MRI"
