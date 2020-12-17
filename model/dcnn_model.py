import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import h5py
import os


def load_data(images, label):
    n = images.shape[0]
    result = []
    for i in range(n):
        result.append([images[i], label[i]])
    return np.array(result, dtype=object)


def shuffle_data(dataset):
    np.random.shuffle(dataset)
    N = dataset.shape[0]
    return np.split(dataset, [int(np.floor(N * 0.7)), int(N)])


def gaussian_weights_init(m):
    """
    Initialize model weights to ~Normal Distribution.
    """
    if 'Conv' in m.__class__.__name__:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def test_data(model, dataset):
    result, num = 0.0, 0
    for images, labels in dataset:
        tensor = torch.tensor(images)
        tensor = torch.reshape(tensor, (1, 1, 48, 48)).to(model.device)
        label = torch.tensor(labels)
        label = label.type(torch.LongTensor).to(model.device)

        pred = model.forward(tensor)
        pred = torch.argmax(pred, axis=1)
        result += torch.sum((pred == label)).item()
        num += 1

    return result / num


class CustomizedCNNModel(nn.Module):
    def __init__(self, alpha, epochs, batch_size, dataset, num_classes=7):
        super(CustomizedCNNModel, self).__init__()
        self.dataset = dataset

        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.epochs_record = []
        self.acc_train = []
        self.acc_test = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # (batch_size, 1, 48, 48) -> (batch_size, 64, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv1.apply(gaussian_weights_init)

        # (batch_size, 64, 24, 24) -> (batch_size, 128, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2.apply(gaussian_weights_init)

        # (batch_size, 128, 12, 12) -> (batch_size, 256, 6, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3.apply(gaussian_weights_init)

        # (batch_size, 256, 6, 6) -> (batch_size, 512, 3, 3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512 * 3 * 3, out_features=1024),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=7),
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.optimizer = optim.SGD(self.parameters(), lr=self.alpha, momentum=0.95, weight_decay=5e-4)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()

    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)

        input = input.view(input.size()[0], -1)
        classes = self.fc(input)

        return classes

    def get_data(self):
        # split dataset into 70% training data and 30% testing data
        print("==============Starting loading data==============")
        self.train_data_loader = self.dataset[0]
        self.data_size = self.dataset[0].shape[0]
        self.test_data_loader = self.dataset[1]
        print(self.train_data_loader.shape)
        print(self.test_data_loader.shape)
        print("=====================Done!=======================")

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_total_loss = 0
            ep_loss = []

            batch_idx = np.random.choice(self.data_size, self.batch_size, replace=False)
            batch = self.train_data_loader[batch_idx]
            for _, (input) in enumerate(batch):
                tensor = torch.tensor(input[0])
                tensor = torch.reshape(tensor, (1, 1, 48, 48)).to(self.device)

                label = torch.tensor([input[1]])
                label = label.type(torch.LongTensor).to(self.device)

                self.optimizer.zero_grad()

                prediction = self.forward(tensor)
                loss = self.loss(prediction, label)

                ep_loss.append(loss.item())
                ep_total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Finish epoch', i + 1, 'total loss %.3f' % ep_total_loss,
                  'loss %.3f' % np.mean(ep_loss))
            self.loss_history.append(ep_total_loss)

            if (i + 1) % 10 == 0:
                train_acc = self._test_data(batch)
                test_acc = self._test_data(self.test_data_loader)
                self.epochs_record.append(i)
                self.acc_train.append(train_acc)
                self.acc_test.append(test_acc)
                print('Training data accuracy after epoch #{}: '.format(i + 1), train_acc)
                print('Testing data accuracy after epoch #{}: '.format(i + 1), test_acc)

    def _test_data(self, dataset):
        self.eval()
        result, num = 0.0, 0
        for images, labels in dataset:
            tensor = torch.tensor(images)
            tensor = torch.reshape(tensor, (1, 1, 48, 48)).to(self.device)
            label = torch.tensor(labels)
            label = label.type(torch.LongTensor).to(self.device)

            pred = self.forward(tensor)
            pred = torch.argmax(pred, axis=1)
            result += torch.sum((pred == label)).item()
            num += 1

        return result / num


if __name__ == "__main__":
    # TRAINING MAIN
    datapath = "/content/drive/MyDrive/CSC420_project/fer2013_data.h5"  # TODO: <- change path

    f = h5py.File(datapath, "r")
    input = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    print(input.shape, labels.shape)

    dataset = load_data(input, labels)

    result = np.array(dataset)

    NUM_EPOCHS = 200
    model = CustomizedCNNModel(alpha=0.0001, epochs=NUM_EPOCHS, batch_size=128,
                               dataset=shuffle_data(result), num_classes=7)
    model._train()

    dst_path = "model_data/cnn_fer2013.pt"
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    torch.save(model, dst_path)

    # Plot
    plt.plot(model.epochs_record, model.acc_train)
    plt.plot(model.epochs_record, model.acc_test)
    plt.legend(['train', 'test'], loc='upper left')
    plt.title("FER2013 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.plot(list(range(1, NUM_EPOCHS+1)), model.loss_history)
    plt.title("FER2013 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
