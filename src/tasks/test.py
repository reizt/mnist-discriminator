import os.path as path
from src.service.dataset import MnistDataset
from src.core.network import Network
from src.service.lab import examine

# Load images and labels for training
datadir = path.join(path.dirname(path.abspath(__file__)), "../data")
dataset = MnistDataset(
    train_images_path=f"{datadir}/train-images",
    train_labels_path=f"{datadir}/train-labels",
    test_images_path=f"{datadir}/test-images",
    test_labels_path=f"{datadir}/test-labels",
)
train_images, train_labels = dataset.load_train()
test_images, test_labels = dataset.load_test()

# Pass the images and labels to train function
data_count, input_size = train_images.shape
network = Network(input_size=input_size)

examine(
    network=network,
    train_input=train_images,
    train_teacher=train_labels,
    test_input=test_images,
    test_answer=test_labels,
    leaning_rate=0.001,
    epochs=10,
    batch_size=100,
)
