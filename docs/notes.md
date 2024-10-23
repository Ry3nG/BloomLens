# Dataset Description: Flowers102

The Flowers102 dataset is a comprehensive collection of flower images designed for image classification tasks. Key characteristics include:

1. **Size and Structure:**
   - Total images: 8,189
   - Classes: 102 different flower categories
   - Splits: Train (1,020 images), Validation (1,020 images), Test (6,149 images)

2. **Class Distribution:**
   - Train and Validation sets: Perfectly balanced with 10 images per class
   - Test set: Imbalanced, ranging from 20 to 238 images per class

3. **Features:**
   - High-quality images of flowers commonly found in the United Kingdom
   - Diverse in terms of scale, pose, and lighting conditions
   - Includes similar categories, challenging fine-grained classification

4. **Implications for Model Training:**
   - Balanced train/validation sets ideal for initial model training
   - Larger, imbalanced test set provides realistic performance evaluation
   - Suitable for developing robust flower classification models

5. **Baseline Model Performance:**

The following tables summarize the test accuracy (%) of various models trained on different dataset sizes:

| Model         | 100% Dataset | 50% Dataset | 25% Dataset | 10% Dataset |
|---------------|--------------|-------------|-------------|-------------|
| alexnet       | 73.96        | 59.89       | 43.72       | 24.27       |
| vgg16         | 81.43        | 66.10       | 48.60       | 26.71       |
| vgg19         | 76.71        | 66.33       | 45.80       | 27.52       |
| resnet18      | 82.06        | 63.31       | 42.03       | 16.94       |
| resnet50      | 85.98        | 71.76       | 49.32       | 27.52       |
| resnet101     | 84.99        | 69.23       | 46.06       | 24.76       |
| densenet121   | 86.94        | 70.43       | 48.28       | 27.36       |
| densenet201   | 88.99        | 73.52       | 47.17       | 22.15       |
| mobilenet_v2  | 86.19        | 73.91       | 49.25       | 19.06       |
| googlenet     | 81.92        | 63.21       | 37.35       | 2.93        |

Key observations:
- DenseNet201 achieves the highest accuracy (88.99%) when trained on the full dataset.
- Performance generally decreases as the dataset size is reduced, as expected.
- Some models (e.g., ResNet50, MobileNetV2) maintain relatively good performance even with reduced data.
- GoogleNet shows a dramatic drop in performance with smaller dataset sizes.

This baseline performance provides a good starting point for further model optimization and fine-tuning.
