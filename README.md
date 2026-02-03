# Data-Efficient Unsupervised Interpolation Without Any Intermediate Frame for 4D Medical Images (CVPR 2024)

[![PyTorch](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![GitHub repo size](https://img.shields.io/github/repo-size/tuhlnaa/UVI-Net-Extended?label=Repo%20size)](https://github.com/tuhlnaa/UVI-Net-Extended)

<br>

## Abstract

[![arXiv](https://img.shields.io/badge/arXiv-2404.01464-B31B1B?logo=arxiv)](https://arxiv.org/abs/2404.01464)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14238626-1682D4?logo=zenodo)](https://doi.org/10.5281/zenodo.14238626)

4D medical images, which represent 3D images with temporal information, are crucial in clinical practice for capturing dynamic changes and monitoring long-term disease progression. However, acquiring 4D medical images poses challenges due to factors such as radiation exposure and imaging duration, necessitating a balance between achieving high temporal resolution and minimizing adverse effects. Given these circumstances, not only is data acquisition challenging, but increasing the frame rate for each dataset also proves difficult. To address this challenge, this paper proposes a simple yet effective **U**nsupervised **V**olumetric **I**nterpolation framework, UVI-Net. This framework facilitates temporal interpolation without the need for any intermediate frames, distinguishing it from the majority of other existing unsupervised methods. Experiments on benchmark datasets demonstrate significant improvements across diverse evaluation metrics compared to unsupervised and supervised baselines. Remarkably, our approach achieves this superior performance even when trained with a dataset as small as one, highlighting its exceptional robustness and efficiency in scenarios with sparse supervision. This positions UVI-Net as a compelling alternative for 4D medical imaging, particularly in settings where data availability is limited.

> **Data-Efficient Unsupervised Interpolation Without Any Intermediate Frame for 4D Medical Images**<br>
> JungEun Kim*, Hangyul Yoon*, Geondo Park, Kyungsu Kim, Eunho Yang <br>

<div align='center'>
    <img src='assets/main_figure.png' width='800px'>
</div>

<br>

### Quantitative Results 
<div align='center'>
    <img src='assets/main_result1.png' width='800px'>
    <img src='assets/main_result2.png' width='800px'>
</div>

<br>

### Qualitative Results (Compared with Top-3 Baselines)
<div align='center'>
  <img src=assets/cardiac.gif width='800px'>
  <img src=assets/lung.gif width='800px'> 
</div>

<br>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/tuhlnaa/UVI-Net-Extended.git
cd modern-uvi-net

# Create and activate conda environment
conda create -n uvinet python=3.11
conda activate uvinet

# Install dependencies
pip install -r requirements.txt
```

<br>


## 💾 Datasets

To use this code, you will need to download the ACDC dataset (7.26 GB) and 4D-Lung dataset (170 GB). You can download the dataset from the [ACDC website](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb), [MICCAI'17 ACDC](https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17) and [4D-Lung website](https://www.cancerimagingarchive.net/collection/4d-lung/).

After downloading the dataset, place the data in the `dataset` directory as follows:
```plain-text
└── dataset
    ├── ACDC
    │   └── database
    │       ├── training
    │       │   ├── patient001
    │       │   │   ├── patient001_4d.nii.gz
    │       │   │   ├── patient001_frame01.nii.gz
    │       │   │   ├── patient001_frame01_gt.nii.gz
    │       │   │   ├── patient001_frame12.nii.gz
    │       │   │   ├── patient001_frame12_gt.nii.gz
    │       │   │   ├── MANDATORY_CITATION.md
    │       │   │   └── Info.cfg
    │       │   ├── patient002
    │       │   │       :
    │       │   └── patient100
    │       ├── testing
    │       │   ├── patient101
    │       │   │       :
    │       │   └── patient150
    │       └── MANDATORY_CITATION.md
    └── 4D-Lung
        ├── 100_HM10395
        │   ├── 09-15-1997-NA-p4-69351
        │   │             :
        │   └── 07-02-2003-NA-p4-14571
        ├── 101_HM10395
        │     :
        └── 119_HM10395
```

<br>

For the 4D-Lung dataset, you need to preprocess the data (e.g. bed removal, min-max scaling, cropping, resizing ...) with the following command:
```bash
.\scripts\preprocess_lung_ct.bat
```

The final data structure should be as follows:
```plain-text
└── dataset
    ├── ACDC
    │   └── (same as above)
    ├── 4D-Lung
    │   └── (same as above)
    └── 4D-Lung_Preprocessed
        ├── 100_0
        │   ├── ct_100_0_frame0.nii.gz
        │   │             :
        │   └── ct_100_0_frame9.nii.gz
        ├── 100_1
        │     :
        └── 119_7
```

<br>

## 🔧 Training

```bash
python train.py --dataset cardiac
python train.py --dataset lung
```

<br>

## 📊 Evaluation

```bash
.\scripts\run_evaluation.bat
```

<br>

## 📝 Citation

This repository is based on the following paper:

```bibtex
@inproceedings{kim2024data,
  title={Data-Efficient Unsupervised Interpolation Without Any Intermediate Frame for 4D Medical Images},
  author={Kim, JungEun and Yoon, Hangyul and Park, Geondo and Kim, Kyungsu and Yang, Eunho},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11353--11364},
  year={2024}
}
```

<br>

## 🎓 Original Work

This is an enhanced implementation of the original [UVI-Net paper](https://github.com/jungeun122333/UVI-Net). While we've modernized the codebase, all credit for the original method goes to the paper authors.

<br>

## 📮 Contact
For questions and feedback:

1. Create an issue in this repository
2. [Google docs](https://docs.google.com/forms/d/e/1FAIpQLSc7obxpa5UXQyDMLE7nssiXzg8Z5qa_kmLBZzqMuslfu8U8vQ/viewform?usp=header)
