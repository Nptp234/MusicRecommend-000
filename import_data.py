import os
import pandas as pd
import re
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import functools

folderName = "dataset"
folderTrainName = "train_spectrogram_image"
folderTestName = "test_spectrogram_image"

#------------
def metadata():
    """Load track metadata into arrays."""
    fileName = f"{folderName}/fma_metadata/tracks.csv"
    track = pd.read_csv(fileName, header=2, low_memory=False)
    trackIdArray = track.iloc[:, 0].values.reshape(-1, 1)
    trackGenreArray = track.iloc[:, 40].values.reshape(-1, 1)
    return trackIdArray, trackGenreArray

#------------
def process_file_train(file_info):
    """Process a single MP3 file into a spectrogram."""
    f, trackIdArray, trackGenreArray, counter = file_info
    track_id = int(re.search(r'fma_small[\\/].*[\\/](.+?).mp3', f).group(1))
    
    try:
        track_index = np.where(trackIdArray == track_id)[0][0]  # Find track_id index faster using np.where
        if str(trackGenreArray[track_index, 0]) != '0':  # Ensure genre is valid
            print(f"Processing: {f}")
            y, sr = librosa.load(f, sr=None)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(mel_spec)

            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.imshow(mel_db, cmap='gray_r', aspect='auto')
            ax.axis('off')

            output_path = f"{folderTrainName}/{counter}_{trackGenreArray[track_index, 0]}.jpg"
            fig.savefig(output_path, bbox_inches=None, pad_inches=0, dpi=100)
            plt.close(fig)
    except IndexError:
        print(f"Track ID {track_id} not found in metadata.")

def process_file_test(file_info):
    print("Process a single MP3 file into a spectrogram.")
    f, count = file_info
    
    try:
        print(f"Processing: {f}")
        y, sr = librosa.load(f, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec)

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(mel_db, cmap='gray_r', aspect='auto')
        ax.axis('off')

        output_path = f"{folderTestName}/{count}.jpg"
        fig.savefig(output_path, bbox_inches=None, pad_inches=0, dpi=100)
        plt.close(fig)
    except FileNotFoundError as e:
        print(f"File not found: {f}, Error: {e}")
    except e:
        print(e)

#-----------
def convertMp3Spectrogram_parallel_Train(fileNames, trackIdArray, trackGenreArray, counter):
    """Convert MP3 files to spectrograms in parallel."""
    file_info = [(f, trackIdArray, trackGenreArray, i + counter) for i, f in enumerate(fileNames)]
    
    with ProcessPoolExecutor() as executor:
        executor.map(process_file_train, file_info)  # Use parallel processing for speed
        
def convertMp3Spectrogram_parallel_Test(fileNames, count):
    """Convert MP3 files to spectrograms in parallel."""
    file_info = [(f, count+i) for i, f in enumerate(fileNames)]
    
    with ProcessPoolExecutor() as executor:
        executor.map(process_file_test, file_info)  # Use parallel processing for speed

#------------
def create_spectrogram_train(verbose=0):
    """Convert MP3 files to spectrograms and save them as images."""
    if os.path.exists('train_spectrogram_image'):
        return
    
    trackIdArray, trackGenreArray = metadata()
    
    folderSample = f"{folderName}/fma_small"
    directions = [os.path.join(folderSample, d) for d in os.listdir(folderSample) if os.path.isdir(os.path.join(folderSample, d))]
    counter = 0
    
    if verbose > 0:
        print("Converting mp3 audio files into mel Spectrograms ...")
    os.makedirs('train_spectrogram_image', exist_ok=True)
    
    for d in directions:
        fileNames = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mp3")]
        convertMp3Spectrogram_parallel_Train(fileNames, trackIdArray, trackGenreArray, counter)
    
    return

#------------
def create_spectrogram_test(verbose=1):
    """Convert MP3 files to spectrograms and save them as images."""
    if os.path.exists('test_spectrogram_image'):
        return
    
    folderSample = f"{folderName}/fma_self_30"
    
    if verbose > 0:
        print("Converting mp3 audio files into mel Spectrograms ...")
        os.makedirs('test_spectrogram_image', exist_ok=True)
        fileNames = [os.path.join(folderSample, f) for f in os.listdir(folderSample) if f.endswith(".mp3")]
        convertMp3Spectrogram_parallel_Test(fileNames, 0)

    return

# create_spectrogram_train()
# create_spectrogram_test()
