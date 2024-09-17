import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets, numpy, random, keras

# print(tensorflow_datasets.list_builders())

# load dataset
DATASET_NAME = "gtzan_music_speech"
(dataset_train_raw, dataset_test_raw), dataset_info = tensorflow_datasets.load(
    name=DATASET_NAME,
    data_dir="tmp",
    with_info=True,
    as_supervised=True,
    split=[tensorflow_datasets.Split.TRAIN, tensorflow_datasets.Split.TEST]
)
print(dataset_info)
NUM_TRAIN = dataset_info.splits['train'].num_examples
NUM_TEST = dataset_info.splits['test'].num_examples
NUM_CLASSES = dataset_info.splits['label'].num_classes