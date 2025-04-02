import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='Training'):
        self.data = pd.read_csv(csv_file)
        # If a 'Usage' column exists, filter by it; otherwise, use all data.
        if 'Usage' in self.data.columns:
            self.data = self.data[self.data['Usage'] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row['emotion'])
        pixels = np.array(list(map(int, row['pixels'].split())), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    # Optionally, add normalization here (e.g., transforms.Normalize(mean, std))
])

# Import the custom dataset class (if you placed it in main.py, it's already available)
from torch.utils.data import Dataset

# Instantiate the dataset
# If you're using train.csv, ensure it’s in your data folder
train_csv = 'data/fer_plus/train.csv'  # adjust the path if needed
from pathlib import Path
if not Path(train_csv).exists():
    print("Error: CSV file not found at", train_csv)
    exit(1)

# Create the dataset filtering only training samples
train_dataset = FER2013Dataset(csv_file=train_csv, transform=transform, usage='Training')

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple CNN model (example)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and move model to the appropriate device
model = SimpleCNN(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (for demonstration, runs for a few epochs)
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # Move images and labels to the GPU if available
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # reset gradients
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Optionally, save the model
torch.save(model.state_dict(), 'models/simple_cnn.pth')



















