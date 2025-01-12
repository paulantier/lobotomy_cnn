import torch
import torch.nn as nn
import torchvision.models as models


class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Freeze the ResNet34 parameters
        for param in resnet34.parameters():
            param.requires_grad = False
        
        # Use only the feature extraction part of ResNet34
        self.features = nn.Sequential(*list(resnet34.children())[:-1])
        
        # Add linear regression layers
        self.regression_layers = nn.Sequential(
            nn.Linear(resnet34.fc.in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression_layers(x)
        x = torch.sigmoid(x) * 100
        return x