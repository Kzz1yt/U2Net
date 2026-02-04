# U2Net

## Model Architecture

### Teacher Model: UNet
- **Backbone Options**: VGG or ResNet50

## Dataset Format

### VOC Format Requirements
- **Input Images**: `.jpg` format
- **Label Images**: `.png` format
- **Pixel Values**:
  - Background: 0
  - Target Classes: 1, 2, ..., num_classes
- **File Structure**:
  ```
  VOCdevkit/
  ├── VOC2007/
  │   ├── ImageSets/
  │   │   └── Segmentation/
  │   │       ├── train.txt
  │   │       ├── val.txt
  │   │       └── trainval.txt
  │   ├── JPEGImages/
  │   └── SegmentationClass/
  ```


## Usage Examples

### Training
1. **Prepare Dataset**: Organize in VOC format
2. **Configure Parameters**: Edit `train.py` or `train_dis.py`
3. **Start Training**:
   ```bash
   python train.py  # Regular UNet training
   python train_dis.py  # Distillation training
   ```

### Inference
1. **Single Image Prediction**:
   ```bash
   python predict.py
   # Enter image path when prompted
   ```

2. **Batch Prediction**:
   ```bash
   # Set mode='dir_predict' in predict.py
   # Configure dir_origin_path and dir_save_path
   python predict.py
   ```


## Troubleshooting

- **Tensor Size Mismatch**: Check feature alignment in FCSF module
- **CUDA Out-of-Memory**: Enable FP16 and reduce batch size
- **Loss Not Converging**: Use pretrained backbone weights
- **Class Imbalance**: Enable focal loss if needed

## References
- Original UNet paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

- Knowledge Distillation: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
