import os, cv2
import modelpointer
import torch
import pandas as pd
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "analysis.csv"
vsi_folder = "VSIs"
output_folder = "output"
model_path = "cnn_model.pt"

def collect_image_paths(root_folder):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.png'):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def analyze_tile(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, pred = torch.max(output, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        pixel_counts = modelpointer.count_pixels(pred)
        return pixel_counts

def load_tile(image):
    padded_image, _, _ = modelpointer.pad_image(image, 256)
    padded_image = ToTensor()(padded_image)
    padded_image = padded_image.unsqueeze(0)
    return padded_image

def process_batch(model, image_paths, csv_path, device, vsi_name, series_num):
    cumulative_counts = {'0': 0, '1': 0, '2': 0}

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tile_tensor = load_tile(image)
        pixel_counts = analyze_tile(model, tile_tensor, device)
        cumulative_counts['0'] += pixel_counts.get(0, 0)
        cumulative_counts['1'] += pixel_counts.get(1, 0)
        cumulative_counts['2'] += pixel_counts.get(2, 0)

    df = pd.DataFrame([{
        'Filename': f"{vsi_name} {series_num}",
        'Count 0': cumulative_counts['0'],
        'Count 1': cumulative_counts['1'],
        'Count 2': cumulative_counts['2']
    }])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"Saved pixel counts to {csv_path}")

def main():
    # Load the model
    model = modelpointer.load_model(model_path, device)

    # Process each VSI file
    for file in os.listdir(vsi_folder):
        if not file.endswith('.vsi'):
            continue
        vsi_file_path = os.path.join(vsi_folder, file)
        base_name, _ = os.path.splitext(file)

        # Process each series subdirectory
        for series_num, series_dir in enumerate(os.listdir(os.path.join(output_folder, base_name)), start=1):
            vsi_output_folder = os.path.join(output_folder, base_name, series_dir)
            image_paths = collect_image_paths(vsi_output_folder)

            # Process the images
            process_batch(model, image_paths, csv_path, device, base_name, series_num)

if __name__ == "__main__":
    main()