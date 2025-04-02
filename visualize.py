import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Needed for softmax

# custom dataset
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

# model architecture
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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- Step 1: Set up device and image transformations -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# ----- Step 2: Create the Test Dataset and DataLoader -----
test_csv = 'data/fer_plus/test.csv'  # update if needed
if not os.path.exists(test_csv):
    print("Test CSV file not found:", test_csv)
    exit(1)

test_dataset = FER2013Dataset(csv_file=test_csv, transform=transform)
# Check if the CSV contains ground truth labels
has_labels = 'emotion' in test_dataset.data.columns
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# ----- Step 3: Load the Saved Model -----
model = SimpleCNN(num_classes=7).to(device)
model_checkpoint = 'models/simple_cnn.pth'
if not os.path.exists(model_checkpoint):
    print("Model checkpoint not found:", model_checkpoint)
    exit(1)
model.load_state_dict(torch.load(model_checkpoint))
model.eval()

# ----- Step 4: Define the emotion mapping -----
emotion_labels = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Sad", 
    5: "Surprise", 
    6: "Neutral"
}

# ----- Step 5: Define a function to visualize predictions with percentages -----
def visualize_predictions(model, data_loader, device, emotion_labels, num_images=8):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 6))
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Get probability distributions via softmax
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            for idx in range(images.size(0)):
                if images_shown == num_images:
                    break
                
                image = images[idx].cpu().squeeze(0)  # Remove batch and channel dimensions
                
                # Get predicted label and its probability percentage
                pred_label = emotion_labels.get(preds[idx].item(), "Unknown")
                pred_prob = probs[idx][preds[idx].item()].item() * 100
                
                # If a valid ground truth exists, display it; otherwise, show only the prediction.
                if has_labels and labels[idx].item() != -1:
                    true_label = emotion_labels.get(labels[idx].item(), "Unknown")
                    title_text = f"True: {true_label}\nPred: {pred_label} ({pred_prob:.0f}%)"
                else:
                    title_text = f"Pred: {pred_label} ({pred_prob:.0f}%)"
                
                plt.subplot(2, num_images // 2, images_shown + 1)
                plt.imshow(image, cmap='gray')
                plt.title(title_text, fontsize=10)
                plt.axis('off')
                images_shown += 1
            if images_shown == num_images:
                break
    plt.tight_layout()
    plt.show()

# ----- Step 6: Visualize predictions on a few test images -----
visualize_predictions(model, test_loader, device, emotion_labels, num_images=8)
