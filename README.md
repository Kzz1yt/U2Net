# UNet Distillation Training Demo

## Project Overview
This project implements knowledge distillation training for UNet segmentation models, focusing on transferring knowledge from a teacher UNet model to a student FastLiteSSNet model using intermediate features.

## Model Architecture

### Teacher Model: UNet
- **Backbone Options**: VGG or ResNet50
- **Input Size**: Multiple of 32 (default: 512x512)
- **Output**: Segmentation map with specified number of classes
- **Intermediate Features**: Returns multi-scale features for distillation

### Student Model: FastLiteSSNet
- **Lightweight Architecture**: Optimized for deployment
- **Fast Cross-Scale Fusion (FCSF)**: Efficient feature fusion module
- **Intermediate Features**: Returns matched features for distillation

## Distillation Training

### Key Components
- **Hybrid Distillation Loss (HDL)**: Combines classification loss, Dice loss, and Cauchy-Schwarz divergence
- **Multi-scale Feature Distillation**: Transfers knowledge across different network layers
- **Feature Alignment**: Ensures teacher and student features have matching dimensions
- **Mixed Precision Training (FP16)**: Reduces memory usage

### Training Process
1. **Freeze Stage**: Backbone frozen, fine-tuning only
2. **Unfreeze Stage**: Full model training with all parameters
3. **Learning Rate Schedule**: Cosine decay or step decay
4. **Batch Size Adjustment**: Based on GPU memory

## Validation & Inference

### Validation Metrics
- **mIoU (mean Intersection over Union)**: Primary evaluation metric
- **Loss Values**: Monitor convergence

### Inference Modes
- **Single Image Prediction**: Interactive input
- **Batch Prediction**: Process entire folder
- **Video Detection**: Real-time processing
- **FPS Testing**: Performance evaluation
- **ONNX Export**: Deployment preparation

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

## Training Configuration

### Key Parameters
- `num_classes`: Number of segmentation classes (excluding background)
- `backbone`: Backbone network (vgg or resnet50)
- `input_shape`: Input image size (e.g., [512, 512])
- `Freeze_Train`: Whether to use freeze training
- `batch_size`: Training batch size (adjust based on GPU memory)
- `optimizer_type`: Optimization algorithm (adam or sgd)
- `lr_decay_type`: Learning rate decay strategy (cos or step)

### Memory Optimization
- Enable FP16 training (`fp16=True`)
- Reduce batch size if CUDA out-of-memory occurs
- Use smaller model width factor for student model

## Dependencies

```bash
pip install -r requirements.txt
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

3. **Video Detection**:
   ```bash
   # Set mode='video' in predict.py
   # Configure video_path
   python predict.py
   ```

4. **FPS Testing**:
   ```bash
   # Set mode='fps' in predict.py
   python predict.py
   ```

5. **ONNX Export**:
   ```bash
   # Set mode='export_onnx' in predict.py
   # Configure onnx_save_path
   python predict.py
   ```

## Model Weights

### Pretrained Weights
- Download from: [Model Zoo](https://github.com/bubbliiiing/unet-pytorch)
- Place in `model_data/` directory

### Training Output
- Weights saved in `logs/` directory
- Loss history saved in `logs/loss_*` directories

## Notes

1. **Dataset Preparation**: Ensure correct VOC format with proper class labels
2. **Memory Management**: Reduce batch size if encountering CUDA out-of-memory errors
3. **Feature Alignment**: Teacher and student models must return compatible intermediate features
4. **Performance Metrics**: Monitor validation loss and mIoU for training progress
5. **Deployment**: Export to ONNX format for deployment in production environments

## Troubleshooting

- **Tensor Size Mismatch**: Check feature alignment in FCSF module
- **CUDA Out-of-Memory**: Enable FP16 and reduce batch size
- **Loss Not Converging**: Use pretrained backbone weights
- **Class Imbalance**: Enable focal loss if needed

## References
- Original UNet paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Knowledge Distillation: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)