import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# Dataset definition returning image, normalized soft labels, and image name.
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None, usage='Training'):
        # Read unified CSV file.
        self.data = pd.read_csv(csv_file)
        
        # If a 'Usage' column exists, filter by the specified usage (e.g., 'Training').
        if 'Usage' in self.data.columns:
            self.data = self.data[self.data['Usage'] == usage]
            
        self.transform = transform
        
        # Define the soft-label columns (ensure these names match your CSV)
        self.label_cols = ['neutral', 'happiness', 'surprise', 'sadness',
                           'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Get the image filename; if not present, create a default name.
        image_name = row['Image name'] if 'Image name' in self.data.columns else f'image_{idx}.png'
        # Extract soft labels from the specified columns.
        labels = row[self.label_cols].values.astype(np.float32)
        label_sum = labels.sum()
        if label_sum > 0:
            labels = labels / label_sum
        else:
            # Fallback: if all counts are zero, use a distribution of zeros.
            labels = np.zeros_like(labels)
        # Convert the label array into a tensor.
        labels = torch.tensor(labels)
        
        # Process the pixel string column:
        # Assuming pixels are space-separated numbers and the image size is 48x48.
        pixels = np.array(list(map(int, row['pixels'].split())), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            
        return image, labels, image_name

# Custom collate function to avoid converting image name strings to tensors.
def custom_collate_fn(batch):
    images, soft_labels, image_names = zip(*batch)
    images = torch.stack(images, 0)
    soft_labels = torch.stack(soft_labels, 0)
    return images, soft_labels, list(image_names)

# Device configuration: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# Define image transformations.
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    # Optional: Add normalization if needed, e.g.:
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

# Path to your unified CSV file (adjust the path as needed).
train_csv = 'data/fer_plus/fer2013new_with_pixels.csv'
if not Path(train_csv).exists():
    print("Error: CSV file not found at", train_csv)
    exit(1)

# Instantiate the dataset filtering only training samples.
train_dataset = FER2013Dataset(csv_file=train_csv, transform=transform, usage='Training')

# Create a DataLoader for training using the custom collate function.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

# Define the neural network model (NineLayerCNN example) for 10 soft label classes.
class NineLayerCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NineLayerCNN, self).__init__()
        
        # Convolutional block with BatchNorm inserted.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adjust fc1 based on the output dimensions after pooling.
        self.fc1 = nn.Linear(256 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = torch.relu(self.bn7(self.conv7(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten.
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits.
        return x

# Instantiate and move model to the appropriate device.
if __name__ == '__main__':
    print("Using device:", device)
    model = NineLayerCNN(num_classes=10).to(device)
    
    # Define the loss criterion using KLDivLoss.
    # We apply torch.log_softmax to the outputs before computing the loss.
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (example for 10 epochs).
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode.
        for images, soft_labels, _ in train_loader:
            images, soft_labels = images.to(device), soft_labels.to(device)
            optimizer.zero_grad()  # Reset gradients.
            
            outputs = model(images)  # Raw logits.
            # Compute loss: compare log_softmax(outputs) with the target soft label distribution.
            loss = criterion(torch.log_softmax(outputs, dim=1), soft_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Optionally, save the trained model.
    checkpoint_path = 'models/new9layer_cnn.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
