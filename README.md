# A light-weight SEM image super-resolution method for real-time nanorobotic manipulation （SRSEMNet）
## A light-weighted SEM Super Resolution Network (SRSEMNet) is conducted by Qinkai Chen, Guangyi Zhang, Liang Fang, Xinjian Fan, Hui Xie, and Zhan Yang. It is implemented by Pytorch. Corresponding email: chenqinkai0426@163.com

## This paper uses a light-weighted scanning electron microscope(SEM) super resolution network for SEM image super-resolution. Also, it has less parameters and faster super-resolution speed which can promote the development of real-time high-precision nanomanipulation system. 

### Abstract
#### This paper proposed a light-weight scanning electron microscope (SEM) image super-resolution method for real-time nanomanipulation. We developed a light-weighted SEM Super Resolution Network (SRSEMNet) based on Lightweight Enhanced Super Resolution Convolutional Neural Network (LESRCNN) with attention mechanisms. The suitability of the network for SEM image super-resolution processing was verified through image comparison, Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM). SR-SEMNet consists of shallow feature extraction layer, deep feature extraction layer, upsampling layer, and image reconstruction layer. The publicly available DIV2K dataset applied as the training sets for the network and the SEM Super Resolution Image Dataset (SSRID) dataset are established as the validation set, thus verifying the performance of SRSEMNet and its super-resolution ability on SEM images. The processing results indicate that SRSEMNet performs better in super-resolution processing of SEM images on SSID evaluation metrics.

## Requirements (Pytorch)  
#### Linux
#### Pytorch 0.41
#### Python 2.7
#### torchvision 
#### openCv for Python

## Training datasets 
#### The training dataset can be downloaded at https://pan.baidu.com/s/1Tbw8wkOLejBganex3lt2aw（secret code：6789）(baiduyun).

## Test datasets 
#### The test dataset of Urban100 can be downloaded at https://pan.baidu.com/s/14fCwo8Umjgr6rGZPIlBAJQ (secret code：1234) (baiduyun).
#### The test dataset of SEM image SSRID can be downloaded at https://pan.baidu.com/s/1MY5rruewxVjV9vZXTg0kFw (secret code：4567) (baiduyun).
