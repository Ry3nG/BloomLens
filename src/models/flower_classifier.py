# src/models/flower_classifier.py

import torch.nn as nn  # type: ignore
from torchvision import models  # type: ignore
import torch  # type: ignore


class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace final FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        """Extract features before the final FC layer"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
