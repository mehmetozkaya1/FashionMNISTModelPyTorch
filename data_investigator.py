from dataset import train_dataset, test_dataset
import matplotlib.pyplot as plt
import numpy as np

class DataInvestigator:
    def __init__(self, train_dataset, test_dataset, n_examples = 1):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_examples = n_examples

    def investigate_data(self):
        self.investigate_data_func()

    def visualize_image(self, ex_datas, ex_data_labels):
        for idx in range(len(ex_datas)):
            plt.imshow(ex_datas[idx].squeeze(), cmap="gray")
            plt.title(self.class_names[ex_data_labels[idx]])
            plt.show()

    def investigate_data_func(self):
        print(f"Train data length : {len(self.train_dataset)}\nTest data length : {len(self.test_dataset)}")

        random_idxs = np.random.randint(0, len(self.train_dataset), self.n_examples)

        ex_datas = []
        ex_data_labels = []

        for index in random_idxs:
            ex_data, ex_data_label= self.train_dataset[index]

            ex_datas.append(ex_data)
            ex_data_labels.append(ex_data_label)

        self.class_names = self.train_dataset.classes
        print(f"Train data number of classes : {len(self.class_names)} : {self.class_names}")
        print(f"Train data class to idx : {self.train_dataset.class_to_idx}")

        print(f"Image shape : {ex_datas[0].shape} [color channels, height, width]")
        print(f"Image Label : {self.class_names[ex_data_labels[0]]}")

        self.visualize_image(ex_datas, ex_data_labels)

dataInvestigator = DataInvestigator(train_dataset, test_dataset, 3)
dataInvestigator.investigate_data()