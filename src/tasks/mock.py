import numpy as np
from src.core.network import Network
from src.service.lab import examine
import matplotlib.pyplot as plt


train_images = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]])
train_labels = np.array([0, 1, 2, 3])
test_images = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 1]])
test_labels = np.array([1, 0, 2, 1])

network = Network(input_size=train_images.shape[1])

train_accuracy_list, test_accuracy_list = examine(
    network=network,
    train_input=train_images,
    train_teacher=train_labels,
    test_input=test_images,
    test_answer=test_labels,
    leaning_rate=0.5,
    epochs=300,
    batch_size=1000,
)

x = np.arange(0, 300)
plt.plot(x, train_accuracy_list, label="train")
plt.plot(x, test_accuracy_list, label="test")
plt.legend()
plt.show()
