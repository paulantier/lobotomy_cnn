import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RegressionCNN(nn.Module):
    def __init__(self, resnet34):
        super(RegressionCNN, self).__init__()
        self.resnet34 = resnet34
        # Use only the feature extraction part of ResNet34
        self.features = nn.Sequential(*list(self.resnet34.children())[:-2])
        
        # Add linear regression layers
        self.regression_layers = nn.Sequential(
            nn.Linear(self.resnet34.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression_layers(x)
        x = torch.sigmoid(x) * 100
        return x