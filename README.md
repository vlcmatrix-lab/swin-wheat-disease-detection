# Data-Efficient Wheat Disease Detection Using Shifted Window Transformer: Enhancing Accuracy, Sustainability, and Global Food Security

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTCE.2025.3582267-blue)](https://doi.org/10.1109/TCE.2025.3582267)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)

**Muhammad Khubaib, Tanzila Kehkashan, Maha Abdelhaq, Muhammad Asghar Khan, Muhammad Zaman, Imran Ashraf, Abdul Rehman, Adnan Akhunzada**

*IEEE Transactions on Consumer Electronics, vol. 71, no. 3, pp. 9006-9020, 2025*

[Paper (IEEE Xplore, open access)](https://ieeexplore.ieee.org/document/11048568) | [Paper (PDF)](paper/swin_wheat_disease_tce2025.pdf)

This repository contains the official code for the paper.

## Summary

The paper detects wheat diseases from images with a Swin Transformer, a vision transformer that applies self-attention inside small shifted windows and builds up features at several scales. This lets it capture both fine local detail and the wider context of a leaf. Three public wheat datasets are combined into nine classes (eight diseases and healthy). Near-duplicate images are removed with a similarity check before the data is split, and the images are equalized, normalized and augmented. Bayesian optimization selects the learning rate, batch size and dropout rate, and Grad-CAM heatmaps show which leaf regions drive each prediction.

The paper reports an accuracy of 99.3%, with a precision of 98.4% and a recall of 98.8%.

## Repository structure

```
swin-wheat-disease-detection/
├── notebooks/
│   └── swin_wheat_disease_detection.ipynb   # main entry point: full pipeline
├── paper/
│   └── swin_wheat_disease_tce2025.pdf       # published version (open access, CC BY 4.0)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

The notebook is designed for Kaggle or Google Colab with a GPU (NVIDIA T4 or better). To run it locally instead, use Python 3.11:

```bash
git clone https://github.com/vlcmatrix-lab/swin-wheat-disease-detection.git
cd swin-wheat-disease-detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The Swin Transformer (Swin-Tiny) is loaded with ImageNet-pretrained weights through PyTorch `timm`. Bayesian optimization uses `optuna`.

## Dataset

The paper combines three public datasets into nine classes: Brown Rust, Crown and Root Rot, Fusarium Head Blight, Healthy, Leaf Rust, Loose Smut, Septoria, Stripe Rust and Yellow Rust.

| Dataset | Images | Source given in the paper |
|---|---|---|
| LWDCD 2020 (Large Wheat Disease Classification Dataset) | 4,321 | LWDCD 2020 |
| Wheat Leaf Disease Dataset | 2,615 | [Kaggle: arindam235/wheat-leaf-disease-dataset](https://www.kaggle.com/datasets/arindam235/wheat-leaf-disease-dataset) |
| Wheat Leaf Dataset | 2,410 | [Mendeley Data: wgd66f8n6h/1](https://data.mendeley.com/datasets/wgd66f8n6h/1) |

The Healthy class is the largest (2,460 images) and Stripe Rust the smallest (208 images).

No data is included in this repository. Please follow each dataset's license terms. The notebook maps folder names to the nine classes automatically:

- **Kaggle:** attach the datasets to the notebook; every image folder under `/kaggle/input` is scanned.
- **Colab or local:** set `DATA_DIRS` in the configuration cell to the folders you downloaded, or list Kaggle datasets in `KAGGLE_SOURCES` to download them with `kagglehub` (needs your Kaggle API credentials). Never commit `kaggle.json` or other credentials.

## Usage

```bash
jupyter notebook notebooks/swin_wheat_disease_detection.ipynb
```

Run all cells in order. The notebook covers:

1. Collecting the images and mapping them to the nine classes
2. Removing near-duplicate images (hash pre-filter, then SSIM), followed by a stratified 70 / 15 / 15 split
3. Preprocessing (resizing to 224x224, histogram equalization, normalization) and augmentation (rotation, flips, brightness, Gaussian noise, elastic deformation)
4. Bayesian optimization of learning rate, batch size and dropout rate (10 iterations, Expected Improvement)
5. Training Swin-Tiny with Adam and cosine annealing for 10 epochs, with early stopping after 5 epochs without improvement
6. Test-set metrics, per-class report and confusion matrix
7. Ablation study, comparison with other models, accuracy versus FLOPs and inference time, and Grad-CAM heatmaps

Figures, metrics and the best model checkpoint are saved to `notebooks/outputs/` (created automatically and excluded from git). The notebook is distributed without saved outputs.

## Results

All values below are reported in the paper.

**Swin Transformer**

| Metric | Value |
|---|---|
| Accuracy | 99.3% |
| Precision | 98.4% |
| Recall | 98.8% |

**Optimal hyperparameters from Bayesian optimization**

| Parameter | Value |
|---|---|
| Learning rate | 0.0001 |
| Batch size | 32 |
| Dropout rate | 0.3 |

**Accuracy of other models reported in the paper**

| Model | Accuracy |
|---|---|
| **Swin Transformer (ours)** | **99.3%** |
| DCNN | 97.88% |
| Data-efficient ViT (DeViT) | 97.2% |
| DenseNet121 | 96.8% |
| ViT | 95.5% |
| ResNet50 | 95% |
| E-MMC | 94.16% |

The full ablation study and state-of-the-art comparison are given in Tables V and VI of the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{khubaib2025wheat,
  author  = {Khubaib, Muhammad and Kehkashan, Tanzila and Abdelhaq, Maha and Khan, Muhammad Asghar and Zaman, Muhammad and Ashraf, Imran and Rehman, Abdul and Akhunzada, Adnan},
  title   = {Data-Efficient Wheat Disease Detection Using Shifted Window Transformer: Enhancing Accuracy, Sustainability, and Global Food Security},
  journal = {IEEE Transactions on Consumer Electronics},
  volume  = {71},
  number  = {3},
  pages   = {9006--9020},
  year    = {2025},
  doi     = {10.1109/TCE.2025.3582267}
}
```

## Paper license

The paper is open access. © 2025 The Authors. It is published by IEEE under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The published version is available at DOI: [10.1109/TCE.2025.3582267](https://doi.org/10.1109/TCE.2025.3582267). The PDF in `paper/` is the published version, shared under that license.

## License

The code in this repository is released under the [MIT License](LICENSE). The paper is licensed separately under CC BY 4.0, as stated above.
