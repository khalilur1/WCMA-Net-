# WCMA-Net: Enhancing Mammographic Cancer Diagnosis Using
Wavelet-Driven Channel-Spatial Mamba Attention

This repository contains the official PyTorch implementation of a wavelet-based multi-scale attention network for binary mammogram classification (benign vs malignant).  
It includes:

- Wavelet + Gaussian notch filter–based preprocessing
- A proposed Wavelet Mamba Attention Network
- An ablation study with multiple architectural variants


1. Project Structure

- `PreProcess_pytorchV2.py`  
  Preprocessing with Gaussian notch filter + wavelet transform and class-weight computation, saving tensors in a `.pt` file for training. :contentReference[oaicite:0]{index=0}  

- `Preproecess_PytorchV1.py`  
  Simpler preprocessing pipeline (same transforms, no class weights in the output). :contentReference[oaicite:1]{index=1}  

- `Proposed_Wav_Mam_Pytorch.py`  
  Implementation of the proposed **MambaAttentionNetwork** with:
  - Multi-scale convolutions (3×3, 5×5, 7×7)
  - Channel-wise and spatial attention modules
  - Training, validation, and final evaluation (accuracy, precision, sensitivity, specificity, F1, AUC)
  - Saving the best model and plotting confusion matrix and ROC curve. :contentReference[oaicite:2]{index=2}  

- `Ablation_Model_Save.py`  
  Implementation of an ablative model where you can selectively enable/disable:
  - Wavelet / denoising flags (for variant naming)
  - Multi-scale convolutions
  - Channel attention
  - Spatial attention  
  Multiple variants are trained and saved as separate `.pth` files. :contentReference[oaicite:3]{index=3}  

---
