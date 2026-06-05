# Medical Image Segmentation Benchmark on Kvasir‑SEG

This repository contains a complete pipeline for training and evaluating medical image segmentation models on the **Kvasir‑SEG** dataset (polyp segmentation). The work includes:

- A flexible PyTorch data loader for Kvasir‑SEG (images, masks, train/val/test splits).
- **Five classic segmentation models** for benchmarking: U‑Net, SegNet, ResUNet, PraNet, and U‑Net++.
- A **lightweight implementation** inspired by the state‑of‑the‑art **ÆMMamba** paper, capturing its core ideas:  
  – Mamba‑like 1D scanning (efficient long‑range dependencies)  
  – Edge‑Aware Module (Sobel operator + gated fusion)  
  – Multi‑scale fusion with boundary‑sensitive decoding  
- CPU‑friendly training (no GPU required) – uses only `torch`, `torchvision`, `PIL`, `tqdm`, `numpy`.

## 📁 Dataset Structure

Download the Kvasir‑SEG dataset from [Simula Dataset Portal](https://datasets.simula.no/kvasir-seg/).


### Install dependencies

```bash
pip install torch torchvision pillow tqdm numpy
```

### Run the code 
```bash
python mambaseg.py
```
