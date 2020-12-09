import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


class CNNByParts(nn.Module):
    def __init__(self, alpha, epochs, batch_size, mouth_dataset, eyes_dataset, num_classes=7):
        super(CNNByParts, self).__init__()
        self.eyes_dataset = eyes_dataset
        self.mouth_dataset = mouth_dataset

        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(256, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()



    def forward(self, batch_data):

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)
        # print("done: ", classes)
        return classes

    def get_data(self):
        # split dataset into 70% training data and 30% testing data
        print("==============Starting loading data==============")
        self.eyes_train_data_loader = self.eyes_dataset[0]
        self.eyes_test_data_loader = self.eyes_dataset[1]
        self.mouth_train_data_loader = self.mouth_dataset[0]
        self.mouth_test_data_loader = self.mouth_dataset[1]
        print("size of eyes training data", self.eyes_train_data_loader.shape)
        print("size of eyes test data", self.eyes_test_data_loader.shape)
        print("size of mouth training data", self.mouth_train_data_loader.shape)
        print("size of mouth test data", self.mouth_test_data_loader.shape)
        print("=====================Done!=======================")

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for _, (input, label) in enumerate(self.train_data_loader):
                input = torch.tensor(input)
                input = torch.reshape(input, (1, 1, 48, 48)).to(self.device)
                label = torch.tensor([label])
                label = label.type(torch.LongTensor).to(self.device)
                # label = torch.reshape(label, (1, 1, 7))
                self.optimizer.zero_grad()
                prediction = self.forward(input)
                # print(prediction)
                loss = self.loss(prediction, label)
                # print("Losssssssssssss:", loss)

                prediction = F.softmax(prediction, dim=1)
                # print("correct!")
                class_ = torch.argmax(prediction, dim=1)
                wrong = torch.where(class_ != label,
                                torch.tensor([1.]).to(self.device),
                                torch.tensor([0.]).to(self.device))

                acc = 1 - torch.sum(wrong)

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Finish epoch', i, 'total loss %.3f' % ep_loss,
                  'accuracy %.3f' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)

    def _test(self):
        self.eval()
        ep_loss = 0
        ep_acc = []
        for _, (input, label) in enumerate(self.test_data_loader):
            input = torch.tensor(input)
            input = torch.reshape(input, (1, 1, 48, 48)).to(self.device)
            label = torch.tensor([label])
            label = label.type(torch.LongTensor).to(self.device)
            prediction = self.forward(input)
            loss = self.loss(prediction, label)
            prediction = F.softmax(prediction, dim=1)
            class_ = torch.argmax(prediction, dim=1)
            wrong = torch.where(class_ != label,
                            torch.tensor([1.]).to(self.device),
                            torch.tensor([0.]).to(self.device))

            acc = 1 - torch.sum(wrong)

            ep_acc.append(acc.item())
            ep_loss += loss.item()

        print('total loss %.3f' % ep_loss,
              'accuracy %.3f' % np.mean(ep_acc))


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("available!")