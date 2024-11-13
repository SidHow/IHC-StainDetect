import matplotlib.pyplot as plt
import torch, cv2, os
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from model import UNet
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load and define the model
def load_model(path, device):
    model = UNet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

#Load and visualize the output of standardized images (256x256 tiles)
#The resize step can be altered to keep resolution so long as the dimensions are divisible by 256.
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (256, 256))  # Resize to match the input size of the model
    image = ToTensor()(image)  # Convert to tensor and normalize to [0, 1]
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def visualize_output(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, dim=1)

        # Visualize the prediction
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axs[0].imshow(to_pil_image(image.squeeze(0).cpu()))
        axs[0].set_title('Original Image')

        # Predicted mask
        axs[1].imshow(pred.squeeze(0).cpu(), cmap='gray')
        axs[1].set_title('Predicted Mask')

        for ax in axs:
            ax.axis('off')

        plt.show()

#Handling of large and arbitrarily sized images.
#The dimensions need to be divisible by 256, so we pad the dimensions so that they are.
def pad_image(image, target_size):
    height, width, _ = image.shape
    pad_height = (target_size - height % target_size) % target_size
    pad_width = (target_size - width % target_size) % target_size
    
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return padded_image, pad_height, pad_width

def load_large_image(image_path, target_size=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    padded_image, pad_height, pad_width = pad_image(image, target_size)
    
    padded_image = ToTensor()(padded_image)  # Convert to tensor and normalize to [0, 1]
    padded_image = padded_image.unsqueeze(0)  # Add batch dimension
    return padded_image, pad_height, pad_width

def visualize_output_padded(model, image, pad_height, pad_width, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, dim=1)

        # Move the prediction to CPU
        pred = pred.squeeze(0).cpu().numpy()

        # Visualize the prediction
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axs[0].imshow(to_pil_image(image.squeeze(0).cpu()))
        axs[0].set_title('Original Image')

        # Predicted mask
        axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('Predicted Mask')

        for ax in axs:
            ax.axis('off')

        plt.show()

#Analysis functions

def count_pixels(pred):
    # Count the number of pixels for each class
    unique, counts = np.unique(pred, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    return pixel_counts

def analyze_image(model, image, pad_height, pad_width, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, dim=1)

        # Move the prediction to CPU and convert to numpy
        pred = pred.squeeze(0).cpu().numpy()

        # Count pixels
        pixel_counts = count_pixels(pred)
        return pixel_counts

def process_batch(model, image_paths, csv_path, device):
    results = []

    for image_path in image_paths:
        # Load and analyze image
        image, pad_height, pad_width = load_large_image(image_path)
        pixel_counts = analyze_image(model, image, pad_height, pad_width, device)

        # Get counts for each class, ensuring keys for missing classes are set to 0
        count_0 = pixel_counts.get(0, 0)
        count_1 = pixel_counts.get(1, 0)
        count_2 = pixel_counts.get(2, 0)

        # Append results
        results.append([os.path.basename(image_path), count_0, count_1, count_2])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results, columns=['Filename', 'Count 0', 'Count 1', 'Count 2'])
    df.to_csv(csv_path, index=False)
    print(f"Saved pixel counts to {csv_path}")

#Example Usage
"""
model = load_model(model_path, device)

image = load_image(image_path)
image, pad_height, pad_width = load_large_image(image_path)

visualize_output(model, image, device)
visualize_output_padded(model, image, pad_height, pad_width, device)

process_batch(model, image_paths, csv_path, device)
"""