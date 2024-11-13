import javabridge
import bioformats
from PIL import Image
import numpy as np
import os
from bioformats import ImageReader, OMEXML

#Requires python3.10 or lower, javabridge is incompatible with python3.11.
#It can be installed for Windows from https://github.com/SchmollerLab/python-javabridge-windows
#This code is not authored by me, though it did work when I used it.
#Likewise, python-bioformats can be installed using --no-deps to skip the faulty javabridge installation.

#Path to your .vsi file
vsi_file_path = 'vsifile.vsi'
vsi_file = os.path.basename(vsi_file_path)
base_name, _ = os.path.splitext(vsi_file)
output_dir = 'output_tiles'

#Process the nth highest resolution shot of each slide.
#Using 0 or 1 will likely fail due to Java memory issues.
size_index = 2


def setup_jvm():
    #Set up and start the Java Virtual Machine.
    javabridge.start_vm(class_path=bioformats.JARS)

def shutdown_jvm():
    #Shutdown the Java Virtual Machine.
    javabridge.kill_vm()

def print_series_metadata(file_path):
    #Print out metadata for each series in the VSI file
    
    # Load metadata using bioformats
    metadata = bioformats.get_omexml_metadata(file_path)
    ome_xml = OMEXML(metadata)  # Create an OMEXML object
    image_count = ome_xml.get_image_count()

    # Print out metadata for each series
    for series_index in range(image_count):
        pixels = ome_xml.image(series_index).Pixels
        print(f"Series {series_index}: SizeX={pixels.SizeX}, SizeY={pixels.SizeY}, SizeC={pixels.SizeC}")


def find_series(file_path, size):
    #VSI files are pyramidal, and have different resolution pictures of different sections.
    #This function sorts them into lists containing the different zoom levels of a given image.
    
    metadata = bioformats.get_omexml_metadata(file_path)
    ome_xml = OMEXML(metadata)
    image_count = ome_xml.get_image_count()

    # Collect series info in a list of tuples (series_index, SizeX)
    series_sizes = []
    for series_index in range(image_count):
        pixels = ome_xml.image(series_index).Pixels
        series_sizes.append((series_index, pixels.SizeX))
        #print(f"Series {series_index}: SizeX={pixels.SizeX}, SizeY={pixels.SizeY}, SizeC={pixels.SizeC}")
    #print(series_sizes)

    i = 0
    samples = []
    currentsamples = []
    current_max_sizex = series_sizes[0][1]

    for unsorted_series in series_sizes:
        index, size_x = unsorted_series

        #Skip label           
        
        if size_x > current_max_sizex:
            if i > 1: 
                samples.append(currentsamples)
                currentsamples = [unsorted_series]
            i += 1
        else:
            if i > 1: currentsamples.append(unsorted_series)
        current_max_sizex = size_x
    
    samples.append(currentsamples)

    #print(samples)

    processed_samples = []
    for sample in samples:
        try:
            processed_samples.append(sample[size][0])
        except: continue

    return processed_samples

def load_vsi_image(file_path, series=0):
    #Load a VSI file using Bio-Formats for a specified series
    
    reader = ImageReader(file_path, perform_init=True)
    reader.rdr.setSeries(series)
    
    metadata = bioformats.get_omexml_metadata(file_path)
    ome_xml = OMEXML(metadata)
    pixels = ome_xml.image(series).Pixels
    
    size_x = pixels.SizeX
    size_y = pixels.SizeY
    size_c = pixels.SizeC  # Number of channels
    image_data = np.zeros((size_y, size_x, size_c), dtype=np.uint8)

    image_data[:, :, :] = reader.read(z=0, t=0, c=None, rescale=False, XYWH=(0, 0, size_x, size_y))
    
    return image_data

def save_tiles(image, tile_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure at least one tile even if image is smaller than tile_size
    y_tiles = max(1, image.shape[0] // tile_size)
    x_tiles = max(1, image.shape[1] // tile_size)

    for i in range(x_tiles):
        for j in range(y_tiles):
            tile = image[j*tile_size:(j+1)*tile_size, i*tile_size:(i+1)*tile_size]
            img = Image.fromarray(tile)
            img.save(os.path.join(output_folder, f'tile_{j}_{i}.png'))

def process_series(vsi_file_path, series_indices):
    for index in series_indices:
        image = load_vsi_image(vsi_file_path, series=index)
        output_dir = f'output/{base_name}/tiles_series_{index}'
        save_tiles(image, 256, output_dir)

setup_jvm()
try:
    #print_series_metadata(vsi_file_path)
    series_indices = find_series(vsi_file_path, size_index)
    #print(series_indices)
    process_series(vsi_file_path, series_indices)
finally:
    shutdown_jvm()
