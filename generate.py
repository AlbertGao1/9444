import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
from tqdm import tqdm  # For the progress bar

# Define class names corresponding to 10 classes.
class_names = ['neutral', 'happiness', 'surprise', 'sadness', 
               'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
from main2 import FER2013Dataset, NineLayerCNN, transform

# Custom collate function to avoid converting strings to tensors.
def custom_collate_fn(batch):
    images, soft_labels, image_names = zip(*batch)
    images = torch.stack(images, 0)
    soft_labels = torch.stack(soft_labels, 0)
    return images, soft_labels, list(image_names)

# File paths (adjust as needed).
csv_file = 'data/fer_plus/fer2013new_with_pixels.csv'
model_path = 'models/new9layer_cnn.pth'
output_csv_file = 'output_predictions.csv'

if not Path(csv_file).exists():
    print("Error: CSV file not found at", csv_file)
    exit(1)

# Create dataset and DataLoader with the custom collate function.
dataset = FER2013Dataset(csv_file, transform=transform, usage='Training')
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define the NineLayerCNN model for 10 classes.

print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = NineLayerCNN(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Generate predictions and store them.
predictions = []
with torch.no_grad():
    for images, targets, image_names in tqdm(data_loader, desc="Generating predictions"):
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        for i, name in enumerate(image_names):
            row = {"Image name": name}
            for j, emotion in enumerate(class_names):
                row[emotion] = probs[i][j]
            predictions.append(row)

df_pred = pd.DataFrame(predictions)
df_pred.to_csv(output_csv_file, index=False, float_format="%.4f")
print(f"Predictions saved to {output_csv_file}")
