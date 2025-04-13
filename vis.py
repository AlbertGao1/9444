import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

# Define class names based on your CSV columns.
class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

# Import the dataset, model, and transform definitions from your main file.
from main2 import FER2013Dataset, NineLayerCNN, transform

# File paths (adjust these paths as needed)
csv_file = 'data/fer_plus/fer2013new_with_pixels.csv'
model_path = 'models/new9layer_cnn.pth'

# Create the dataset and randomly select one sample for visualization.
dataset = FER2013Dataset(csv_file, transform=transform, usage='Training')
sample_idx = random.randint(0, len(dataset) - 1)
# Unpack the extra value (image name) with an underscore.
sample_image, target_soft, _ = dataset[sample_idx]

# Unsqueeze image to add batch dimension for the model.
input_image = sample_image.unsqueeze(0)

# Load the trained model and set it to evaluation mode.
model = NineLayerCNN(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    output_logits = model(input_image)
    # Convert logits to probabilities.
    predicted_probs = torch.softmax(output_logits, dim=1).squeeze(0).cpu().numpy()

# Convert target soft labels to numpy and compute percentages.
target_probs = target_soft.cpu().numpy()
predicted_perc = predicted_probs * 100
target_perc = target_probs * 100

# Create subplots: one for the image and one for the bar chart.
fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

# Display the image (convert tensor back to numpy array and squeeze channel dimension).
image_np = sample_image.squeeze().cpu().numpy()
ax_img.imshow(image_np, cmap='gray')
ax_img.axis('off')
ax_img.set_title("Input Image")

# Plot the bar chart.
x = np.arange(len(class_names))
width = 0.35
rects1 = ax_bar.bar(x - width/2, target_perc, width, label='Target')
rects2 = ax_bar.bar(x + width/2, predicted_perc, width, label='Predicted')
ax_bar.set_ylabel('Probability (%)')
ax_bar.set_title('Emotion Distribution')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(class_names, rotation=45)
ax_bar.legend()

def autolabel(rects, ax):
    """Attach a small text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

autolabel(rects1, ax_bar)
autolabel(rects2, ax_bar)

plt.tight_layout()
plt.show()
