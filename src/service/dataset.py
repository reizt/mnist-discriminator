import numpy as np
from dataclasses import dataclass


@dataclass
class MnistDataset:
    train_images_path: str
    train_labels_path: str
    test_images_path: str
    test_labels_path: str

    def __read_images(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            # Read header info
            magic, data_num, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype(">i4"))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(data_num, rows * cols)

    def __read_labels(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            # Read header info
            magic, labels_count = np.frombuffer(f.read(8), dtype=np.dtype(">i4"))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data

    def load_train(self) -> tuple[np.ndarray, np.ndarray]:
        images = self.__read_images(self.train_images_path)
        labels = self.__read_labels(self.train_labels_path)

        return images, labels

    def load_test(self) -> tuple[np.ndarray, np.ndarray]:
        images = self.__read_images(self.test_images_path)
        labels = self.__read_labels(self.test_labels_path)

        return images, labels
