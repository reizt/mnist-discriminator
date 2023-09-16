import numpy as np
from src.core.network import Network


def examine(
    *,
    network: Network,
    train_input: np.ndarray,
    train_teacher: np.ndarray,
    test_input: np.ndarray,
    test_answer: np.ndarray,
    leaning_rate: float,
    epochs: int,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    train_accuracy = network.test_accuracy(train_input, train_teacher)
    test_accuracy = network.test_accuracy(test_input, test_answer)
    print(f"initial: train_accuracy: {round(train_accuracy*100, 0)}%, test_accuracy: {round(test_accuracy*100, 0)}%")

    if epochs < 1:
        raise ValueError("epochs must be larger than 1")
    if batch_size < 1:
        raise ValueError("batch_size must be larger than 1")

    train_size = train_input.shape[0]
    batch_count = max(train_size // batch_size, 1)

    train_accuracy_list: list[float] = []
    test_accuracy_list: list[float] = []
    for i in range(epochs):
        for j in range(batch_count):
            mask = np.random.choice(train_size, batch_size)
            network.train(train_input[mask], train_teacher[mask], leaning_rate=leaning_rate)
        train_accuracy = network.test_accuracy(train_input, train_teacher)
        test_accuracy = network.test_accuracy(test_input, test_answer)
        print(f"epoch {i}: train_accuracy: {round(train_accuracy*100, 0)}%, test_accuracy: {round(test_accuracy*100, 0)}%")
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

    return train_accuracy_list, test_accuracy_list
