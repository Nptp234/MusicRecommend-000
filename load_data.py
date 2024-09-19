import os, re, cv2, keras
import numpy as np
from import_data import create_spectrogram_test, create_spectrogram_train
from slice_spectrogram import slice_spect
from sklearn.model_selection import train_test_split

slice_test_folder = "test_slice_image"
slice_train_folder = "train_slice_image"
spectrogram_test_folder = "test_spectrogram_image"
spectrogram_train_folder = "train_spectrogram_image"

TARGET_SIZE = (128, 128)

genre = {
    "Hip-Hop": 0,
    "International": 1,
    "Electronic": 2,
    "Folk" : 3,
    "Experimental": 4,
    "Rock": 5,
    "Pop": 6,
    "Instrumental": 7
}

#----------
def convert_labels2hotvector(train_x, test_x, train_y, test_y):
    folder_save = "training_data"
    np_utils = keras.utils
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)
    if os.path.exists(folder_save):
        np.load(f"{folder_save}/train_x.npy")
        np.load(f"{folder_save}/train_y.npy")
        np.load(f"{folder_save}/test_x.npy")
        np.load(f"{folder_save}/test_y.npy")
        return train_x, train_y, test_x, test_y
    
    os.makedirs(folder_save)
    np.save(f"{folder_save}/train_x.npy", train_x)
    np.save(f"{folder_save}/train_y.npy", train_y)
    np.save(f"{folder_save}/test_x.npy", test_x)
    np.save(f"{folder_save}/test_y.npy", test_y)
    return train_x, train_y, test_x, test_y

#-----------
def load_dataset_color(file_path):
    # Read the image using OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image was loaded correctly
    if img is None:
        raise ValueError(f"Image not loaded correctly from path: {file_path}")
    
    img_resized = cv2.resize(img, TARGET_SIZE)
    
    # Filter out None values from image_all and label_all
    # resized_images = [cv2.resize(img, TARGET_SIZE) if img.shape[:2] != TARGET_SIZE else img for img in filtered_images]
    
    # Check the number of channels
    if len(img_resized.shape) == 2:  # Grayscale image
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif len(img_resized.shape) == 3 and img_resized.shape[2] == 3:  # RGB image
        img_rgb = img_resized
    else:
        raise ValueError("Unsupported image format.")
    
    # Convert BGR (OpenCV default) to RGB
    # img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    return img_rgb

#-----------
def load_dataset_test(dataset_size = 1.0):
    if not os.path.exists(slice_test_folder) and not os.path.exists(spectrogram_test_folder):
        create_spectrogram_test()
        slice_spect(slice_test_folder, spectrogram_test_folder)
    
    print("Compiling Testing Sets ...")
    file_names = [os.path.join(slice_test_folder, f) for f in os.listdir(slice_test_folder) if f.endswith(".jpg")]
    image = []
    label = []
    for f in file_names:
        image.append(load_dataset_color(f))
        label.append(re.search('.*_(.+?)\.jpg', f).group(1))
    images = np.array(image)
        
    return images, label
#-----------------------------------------------------------
def load_dataset_train(dataset_size = 1.0):
    if not os.path.exists(slice_train_folder) and not os.path.exists(spectrogram_train_folder):
        create_spectrogram_train()
        slice_spect(slice_train_folder, spectrogram_train_folder)
    
    print("Compiling Training Sets ...")
    file_names = [os.path.join(slice_train_folder, f) for f in os.listdir(slice_train_folder) if f.endswith(".jpg")]
    image_all = [None]*(len(file_names))
    label_all = [None]*(len(file_names))
    for f in file_names:
        index = int(f[f.rfind('\\') + 1])
        genre_variable = re.search('.*_(.+?)\.jpg', f).group(1)
        print(f"Loading {index}_{genre_variable}...")
        
        
        image_all[index] = load_dataset_color(f)
        label_all[index] = genre[genre_variable]
    
    if dataset_size==1.0:
        images = image_all
        labels = label_all
    else:
        count_max = int(len(image_all)*dataset_size/8.0)
        count_array = [0,0,0,0,0,0,0,0]
        images = []
        labels = []
        for i in range(0,len(image_all)):
            if (count_array[label_all[i]]<count_max):
                images.append(image_all[i])
                labels.append(label_all[i])
                count_array[label_all[i]]+=1
                
    filtered_images = [img for img in images if img is not None]
    filtered_labels = [label for label in labels if label is not None]
    images=np.array(filtered_images)
    labels=np.array(filtered_labels)
    labels = labels.reshape(labels.shape[0], 1)
    
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.05, shuffle=True)
    genre_new = {value: key for key, value in genre.items()}
    n_classes = len(genre)
    
    train_x, train_y, test_x, test_y = convert_labels2hotvector(train_x, test_x, train_y, test_y)
    
    return train_x, train_y, test_x, test_y, n_classes, genre_new
    
# load_dataset_test()

    
        