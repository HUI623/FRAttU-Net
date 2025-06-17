# FRAttU-Net
Official PyTorch implementation of the paper [Feature refinement attention U-net for enhanced liver segmentation](https://www.sciencedirect.com/science/article/pii/S0952197625013764).

# Architecture

![FRAttU-Net](https://github.com/HUI623/FRAttU-Net/blob/main/FRAttU-Net%20architecture.png)
## 1.Requirements
```bash
# Environment Setup  
Python 3.7.13  
CUDA 11.1  
Pytorch 1.10.1  
torchvision 0.11.2
```
## 2. Installation
```bash
git clone https://github.com/HUI623/FRAttU-Net  
cd FRAttU-Net  
sh install.sh
```
## 3. Data Preprocessing
### Only use the liver and 20 slices above and below the liver as training samples.
```bash
python preprocessing_nii.py
```
### Convert nii to png.
```bash
python preprocessing_png.py
```
## 4. Training
### Modify training file name.
```bash
utils/dataset.py
```
### Training
```bash
python main.py
```
## Testing
```bash
python test_nii_png.py
python test_png_png.py
```

## Acknowledgements
Thanks to [SAR-U-Net](https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation), [UGS-Net](https://github.com/yanghan-yh/UGS-Net), and [Gabor CNN](https://github.com/jxgu1016/Gabor_CNN_PyTorch) for their outstanding work.
