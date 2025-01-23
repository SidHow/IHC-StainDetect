from PIL import Image
import os

#Requires pillow
#pip install pillow

#########################################
#PARAMS
#########################################

#directory = "C:/Directory" #<-- Uncomment this if you want to specify somewhere 
directory = os.getcwd() #<-- Directory we're currently working in

#########################################
#BINNING VISUALIZATION FUNCTIONS
#########################################

def rg_swap(red, green, blue):
    colour_code = (green, red, blue) #Swap RG
    return colour_code

def create_modded_image(image_path, output_path):
    img = Image.open(image_path)
    width, height = img.size
    new_img = Image.new("RGB", (width, height))

    pixels = img.load()
    new_pixels = new_img.load()

    for x in range(width):
        for y in range(height):
            pixel = pixels[x, y]
            if len(pixel) == 3:
                r, g, b = pixels[x, y]
            else: r, g, b, i = pixels[x, y]
            new_pixels[x, y] = rg_swap(r, g, b)

    new_img.save(output_path)
    return new_img

def batch_process(file_directory):
    files = os.listdir(file_directory)
    files = [f for f in files if (f.lower().endswith('.png') or f.lower().endswith('.bmp') or f.lower().endswith('.jpg') or f.lower().endswith('.tif'))] #Could probably support more types, would need to test
    output_dir = os.path.join(file_directory, "output")
    os.makedirs(output_dir, exist_ok=True)
    for file in files:
        path = os.path.join(file_directory, file)
        output = os.path.join(output_dir, file)
        create_modded_image(path, output)

#########################################
#MAIN
#########################################

#create_modded_image("test.png", "test_processed.png") #<-- Uncomment this for one-shots
batch_process(directory)