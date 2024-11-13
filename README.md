# IHC-StainDetect

This code was used to train a U-Net computer vision model to detect staining levels in immunohistochemistry data.
During testing it was found to perform poorly with out-of-band data, so it's recommended to train it using masks generated from your own samples.

## Usage

### Optional:
Tile_Extractor.py can be used to extract .vsi files into .png tiles.
It's recommended to install the CUDA toolkit (if you have a GPU that supports it), as this will greatly improve training and analysis speed.

### 1. Prepare Masks and image loader
Initially, you will need to manually label training data using grayscale masks. Use 256x256 images, and an intensity of 75 for stained and 150 for unstained (configurable in imageloader.py).
imageloader.py will load training and validation ground truth and masks from the directories train_image_dir and val_image_dir. Depending on how much data you provide, you may need to adjust the training and validation batch sizes.

### 2. Train the Model
Once your masks have been prepared, run training.py. This will train your model on your data while providing loss, accuracy, and precision values for each epoch. By default, the epoch with the lowest validation loss will be saved.

### 3. Manual Validation
modelpointer.py contains some supporting functions that visualize the model's predictions and generate masks. Run functions such as visualize_output() against different areas to ensure the model is behaving as intended - if not, consider creating masks of the offending areas and adding them to the training and validation data.

### 4. Analyze Data
Once the model has been refined and tested, batch_analyze.py can be used to analyze data and output pixel counts for each sample. By default, it assumes that the data consists of .png files extracted from .vsi files, so it will need to be altered if your use-case is different.