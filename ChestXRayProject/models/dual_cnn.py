# models/dual_cnn.py
import torch
import torch.nn as nn
import torchvision.models as models

class DualCNN(nn.Module):
    def __init__(self):
        super(DualCNN, self).__init__()
        self.stream1 = models.resnet18(pretrained=True)
        self.stream1.fc = nn.Identity()

        self.stream2 = models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x1 = self.stream1(x)                        # ResNet
        x2 = self.stream2(x)                        # DenseNet
        x2 = self.avgpool(x2).view(x2.size(0), -1)  # Flatten
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)
