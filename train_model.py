import torch
from torch.utils.data import DataLoader
from PawpularityDataset import PawpularityDataset
from RegressionCNN import RegressionCNN
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 20

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = PawpularityDataset(csv_file='train.csv', img_dir='train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = PawpularityDataset(csv_file='test.csv', img_dir='test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = RegressionCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, score in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        predicted_score = model(images)
        loss = criterion(predicted_score, score)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# Save the model
torch.save(model.state_dict(), 'regression_cnn.pth')
print("Model saved as regression_cnn.pth")