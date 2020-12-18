import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


class CNNByParts(nn.Module):
    def __init__(self, alpha, epochs, batch_size, dataset, num_classes=7):
        super(CNNByParts, self).__init__()
        self.dataset = dataset

        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(self.calc_input_dims(), self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()

    def calc_input_dims(self):
        batch_data = torch.zeros((1, 1, 32, 64)) # batch_size, channel, row, col
        batch_data = self.cnn_layer1(batch_data)
        batch_data = self.cnn_layer2(batch_data)

        return int(np.prod(batch_data.size())) * 2 + 384

    def forward(self, eye, mouth, hog_eyes, hog_mouth):
        eye = self.cnn_layer1(eye)
        eye = self.cnn_layer2(eye)
        eye = eye.view(eye.size()[0], -1)

        mouth = self.cnn_layer3(mouth)
        mouth = self.cnn_layer4(mouth)
        mouth = mouth.view(mouth.size()[0], -1)

        hog_eyes = hog_eyes.view(hog_eyes.size()[0], -1)
        hog_mouth = hog_mouth.view(hog_mouth.size()[0], -1)

        concat = torch.cat((eye, mouth, hog_eyes, hog_mouth), dim=1)
        classes = self.fc1(concat)
        return classes

    def get_data(self):
        # split dataset into 70% training data and 30% testing data
        print("==============Starting loading data==============")
        self.train_data_loader = self.dataset[0]
        self.test_data_loader = self.dataset[1]
        print(self.train_data_loader.shape)
        print(self.test_data_loader.shape)
        # print("size of eyes training data", self.eyes_train_data_loader.shape)
        # print("size of eyes test data", self.eyes_test_data_loader.shape)
        # print("size of mouth training data", self.mouth_train_data_loader.shape)
        # print("size of mouth test data", self.mouth_test_data_loader.shape)
        print("=====================Done!=======================")

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for _, (eyes, mouths, hogs_for_eye, hogs_for_mouth) in enumerate(self.train_data_loader):
                # print(eyes.shape, mouth.shape)
                # break
                eye = torch.tensor(eyes[0])
                eye = torch.reshape(eye, (1, 1, 32, 64)).to(self.device)
                mouth = torch.tensor(mouths[0])
                mouth = torch.reshape(mouth, (1, 1, 32, 64)).to(self.device)

                # ----------------------------
                eyes_hog = torch.tensor(hogs_for_eye[0])
                eyes_hog = torch.reshape(eyes_hog, (1, 1, 4, 8, 6)).to(self.device)
                mouth_hog = torch.tensor(hogs_for_mouth[0])
                mouth_hog = torch.reshape(mouth_hog, (1, 1, 4, 8, 6)).to(self.device)
                # ----------------------------

                label = torch.tensor([eyes[1]])
                label = label.type(torch.LongTensor).to(self.device)
                # label = torch.reshape(label, (1, 1, 7))
                self.optimizer.zero_grad()
                prediction = self.forward(eye, mouth, eyes_hog, mouth_hog)
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
        for _, (eyes, mouths, hogs_for_eye, hogs_for_mouth) in enumerate(self.test_data_loader):
            eye = torch.tensor(eyes[0])
            eye = torch.reshape(eye, (1, 1, 32, 64)).to(self.device)
            mouth = torch.tensor(mouths[0])
            mouth = torch.reshape(mouth, (1, 1, 32, 64)).to(self.device)

            eyes_hog = torch.tensor(hogs_for_eye[0])
            eyes_hog = torch.reshape(eyes_hog, (1, 1, 4, 8, 6)).to(self.device)
            mouth_hog = torch.tensor(hogs_for_mouth[0])
            mouth_hog = torch.reshape(mouth_hog, (1, 1, 4, 8, 6)).to(self.device)

            label = torch.tensor([eyes[1]])
            label = label.type(torch.LongTensor).to(self.device)
            prediction = self.forward(eye, mouth, eyes_hog, mouth_hog)
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