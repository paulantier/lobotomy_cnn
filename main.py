import torch
from torch.utils.data import DataLoader
from PawpularityDataset import PawpularityDataset
from RegressionCNN import RegressionCNN
from torchvision import transforms

import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test dataset
test_dataset = PawpularityDataset(csv_file='test.csv', img_dir='dataset/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = RegressionCNN().to(device)
model.load_state_dict(torch.load('regression_cnn.pth'))
model.eval()

# Function to plot images and their scores
def plot_images(images, scores):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        score = scores[i].item()
        axes[i].imshow(image)
        axes[i].set_title(f'Score: {score:.2f}')
        axes[i].axis('off')
    plt.show()

# Get 4 images and their scores from the test set
images, scores = next(iter(test_loader))
images, scores = images[:4].to(device), scores[:4].to(device)
with torch.no_grad():
    predicted_scores = model(images)

# Plot the images and their predicted scores
plot_images(images, predicted_scores)