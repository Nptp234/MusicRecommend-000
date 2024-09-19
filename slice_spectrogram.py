"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""

import os, re
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

train_folder_img = "train_spectrogram_image"
test_folder_img = "test_spectrogram_image"

test_path = "test_slice_image"
train_path = "train_slice_image"

#-------
def process_spectrogram(file_info):
    f,counter, path = file_info
    if path == train_path:
        variable = f.split("_")[-2].split("image\\")[1]
    else:
        variable = f.split("_")[-1].split("image\\")[1].split(".jpg")[0]
    print(f)
    img = Image.open(f)
    sub_sample_s = 128
    width, height = img.size
    number_of_sample = width//sub_sample_s
    for i in range(number_of_sample):
        start = i*sub_sample_s
        img_temporary = img.crop((start, 0., start + sub_sample_s, sub_sample_s))
        img_temporary.save(f"{path}/"+str(counter)+"_"+variable+".jpg")
        counter = counter + 1

#-------
def slice_spectrogram(file_names, counter, path):
    file_info = [(f, counter+i, path) for i, f in enumerate(file_names)]
    with ProcessPoolExecutor() as executor:
        executor.map(process_spectrogram, file_info)

#-------
def slice_spect(path, folder_img, verbose = 1):
    if os.path.exists(path):
        return
    
    os.makedirs(path)
    
    file_names = [os.path.join(folder_img, f) for f in os.listdir(folder_img) if f.endswith(".jpg")]
    counter = 0
    if verbose>0:
        print("Slicing Spectograms ...")
        
    slice_spectrogram(file_names, counter, path)

#-------
slice_spect(train_path, train_folder_img)