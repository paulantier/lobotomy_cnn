import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        # Use only the feature extraction part of ResNet34
        self.features = nn.Sequential(*list(resnet34.children())[:-2])
        
        # Add linear regression layers
        self.regression_layers = nn.Sequential(
            nn.Linear(resnet34.fc.in_features, 512),
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