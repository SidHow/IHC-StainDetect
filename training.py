import torch
from losses import mIoULoss
from model import UNet
import imageloader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

#Parameters
num_epochs = 400
patience = 30
checkpoint_path = "training_checkpoint.pt"

#Init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=3).to(device)
criterion = mIoULoss(n_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
best_val_loss = 100

#Define supporting functions
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_precision = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_acc += pixel_accuracy(outputs, masks)
            val_precision += precision(outputs, masks)
    
    return val_loss / len(val_loader), val_acc / len(val_loader), val_precision / len(val_loader)

def pixel_accuracy(outputs, masks):
    _, preds = torch.max(outputs, dim=1)  #Get the class with the highest probability
    correct = torch.eq(preds, masks).sum().item()  #Count correct predictions
    total = torch.numel(masks)  #Total number of pixels
    return correct / total

def precision(outputs, masks):
    _, preds = torch.max(outputs, dim=1)
    true_positives = torch.sum((preds == 1) & (masks == 1)).item()
    predicted_positives = torch.sum(preds == 1).item()
    
    if predicted_positives == 0:
        return 0
    else:
        return true_positives / predicted_positives

def visualize_predictions(model, data_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            # Visualize the first `num_samples` predictions
            for j in range(min(num_samples, images.size(0))):
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axs[0].imshow(to_pil_image(images[j].cpu()))
                axs[0].set_title('Original Image')

                # Ground truth mask
                axs[1].imshow(masks[j].cpu(), cmap='gray')
                axs[1].set_title('Ground Truth Mask')

                # Predicted mask
                axs[2].imshow(preds[j].cpu(), cmap='gray')
                axs[2].set_title('Predicted Mask')

                for ax in axs:
                    ax.axis('off')
                
                plt.show()

            if i >= num_samples // images.size(0):
                break

#Main training function

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0

    for images, masks in imageloader.train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += pixel_accuracy(outputs, masks)
        epoch_precision += precision(outputs, masks)
    
    epoch_loss /= len(imageloader.train_loader)
    epoch_acc /= len(imageloader.train_loader)
    epoch_precision /= len(imageloader.train_loader)
    val_loss, val_acc, val_precision = validate(model, criterion, imageloader.val_loader, device)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {round(epoch_loss,4)}, Training Accuracy: {round(epoch_acc,4)}, Training Precision: {round(epoch_precision,4)}, Validation Loss: {round(val_loss,4)}, Validation Accuracy: {round(val_acc,4)}, Validation Precision: {round(val_precision,4)}')

    #Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, checkpoint_path)
        print(f"Validation loss improved. Model saved at {checkpoint_path}.")
        counter = 0
    else:
        counter += 1
        print(f"No improvement in validation loss for {counter} epochs.")

    #Early stopping
    if counter >= patience:
        print(f"Early stopping triggered after {patience} epochs of no improvement.")
        break

# Load the best model
#model = load_model(model, checkpoint_path)

# Visualize predictions
#visualize_predictions(model, imageloader.val_loader, device)