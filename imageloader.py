from torch.utils.data import Dataset, DataLoader
import torch
import cv2, os

train_image_dir = 'cell_training'
val_image_dir = 'cell_validation'

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Normalize mask values
        mask = torch.tensor(mask, dtype=torch.long)
        mask[mask == 75] = 1
        mask[mask == 150] = 2

        return image, mask
    

def get_image_and_mask_paths(image_dir, validation_dir):
    train_image_paths = []
    train_mask_paths = []
    val_image_paths = []
    val_mask_paths = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png") and not filename.endswith("_out.png"):
            train_image_paths.append(os.path.join(image_dir, filename))
            mask_filename = filename.replace(".png", "_out.png")
            train_mask_paths.append(os.path.join(image_dir, mask_filename))
    
    for filename in os.listdir(validation_dir):
        if filename.endswith(".png") and not filename.endswith("_out.png"):
            val_image_paths.append(os.path.join(validation_dir, filename))
            mask_filename = filename.replace(".png", "_out.png")
            val_mask_paths.append(os.path.join(validation_dir, mask_filename))

    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths

train_image_paths, train_mask_paths, val_image_paths, val_mask_paths = get_image_and_mask_paths(train_image_dir, val_image_dir)

train_dataset = SegmentationDataset(train_image_paths, train_mask_paths)
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)

val_dataset = SegmentationDataset(val_image_paths, val_mask_paths)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)