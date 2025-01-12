import torch
from torch.utils.data import DataLoader
from PawpularityDataset import PawpularityDataset
from RegressionCNN import RegressionCNN
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 20

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = PawpularityDataset(csv_file='train.csv', img_dir='dataset/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = PawpularityDataset(csv_file='test.csv', img_dir='dataset/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = RegressionCNN().to(device)
#criterion = nn.MSELoss().to(device)
criterion = nn.SmoothL1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (images, score) in enumerate(train_loader_tqdm):
        images, score = images.to(device), score.to(device)
        optimizer.zero_grad()
        predicted_score = model(images)
        loss = criterion(predicted_score.view(-1), score.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/(batch_idx + 1))
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}')
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels.float())
            test_loss += loss.item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# Save the model
torch.save(model.state_dict(), 'regression_cnn.pth')
print("Model saved as regression_cnn.pth")