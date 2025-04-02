import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image

# ----- Step 1: Define your custom dataset and model classes -----
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Check if the 'emotion' column exists in the CSV
        if 'emotion' in self.data.columns:
            label = int(row['emotion'])
        else:
            label = -1  # default label when not available
        pixels = np.array(list(map(int, row['pixels'].split())), dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

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

# ----- Step 2: Set up device and image transformations -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# ----- Step 3: Create the Test Dataset and DataLoader -----
test_csv = 'data/fer_plus/test.csv'  # adjust path if needed
if not os.path.exists(test_csv):
    print("Test CSV file not found:", test_csv)
    exit(1)

test_dataset = FER2013Dataset(csv_file=test_csv, transform=transform)
# Check if the CSV contains ground truth labels
has_labels = 'emotion' in test_dataset.data.columns
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----- Step 4: Load the Saved Model -----
model = SimpleCNN(num_classes=7).to(device)
model_checkpoint = 'models/simple_cnn.pth'
if not os.path.exists(model_checkpoint):
    print("Model checkpoint not found:", model_checkpoint)
    exit(1)

model.load_state_dict(torch.load(model_checkpoint))
model.eval()

# ----- Step 5: Run Inference and Compute Accuracy (if labels are available) -----
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if has_labels:
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

if has_labels and total > 0:
    accuracy = 100 * correct / total
    print("Test Accuracy: {:.2f}%".format(accuracy))
else:
    print("Test CSV does not contain ground truth labels. Only inference was performed.")
