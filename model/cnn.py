import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, alpha, epochs, batch_size, dataset, num_classes=7):
        super(CNN, self).__init__()
        self.dataset = dataset # format list of [input, label] tuple

        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []

        self.conv1 = nn.Conv2d(1, 48, 3)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 48, 3)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 48, 3)
        self.bn3 = nn.BatchNorm2d(48)
        self.maxpool1 = nn.MaxPool2d(2) # decrease dimensionality

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.loss = nn.CrossEntropyLoss()
        self.get_data()

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, 48, 48))
        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)

        return int(np.prod(batch_data.size()))

    def forward(self, batch_data):

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)
        # print("done: ", classes)
        return classes

    def get_data(self):
        # split dataset into 80% training data and 20% testing data
        print("==============Starting loading data==============")
        # shuffle
        # self.dataset = np.random.shuffle(self.dataset)
        np.random.shuffle(self.dataset)
        len = self.dataset.shape[0]
        spliter = np.split(self.dataset, [int(np.floor(len * 0.7)), int(len)])

        self.train_data_loader = spliter[0]
        self.test_data_loader = spliter[1]
        print("size of training data", self.train_data_loader.shape)
        print("size of test data", self.test_data_loader.shape)
        print("=====================Done!=======================")
        
         
    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for _, (input, label) in enumerate(self.train_data_loader):
                input = T.tensor(input)
                input = T.reshape(input, (1, 1, 48, 48))
                label = T.tensor([label])
                label = label.type(T.LongTensor)
                # label = T.reshape(label, (1, 1, 7))
                self.optimizer.zero_grad()
                prediction = self.forward(input)
                # print(prediction)
                loss = self.loss(prediction, label)
                # print("Losssssssssssss:", loss)

                prediction = F.softmax(prediction, dim=1)
                # print("correct!")
                class_ = T.argmax(prediction, dim=1)
                wrong = T.where(class_ != label, 
                                T.tensor([1.]),
                                T.tensor([0.]))
                
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Finish epoch', i, 'total loss %.3f' % ep_loss,
                    'accuracy %.3f' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)
    
    def _test(self):
        ep_loss = 0
        ep_acc = []
        for _, (input, label) in enumerate(self.test_data_loader):
            input = T.tensor(input)
            input = T.reshape(input, (1, 1, 48, 48))
            label = T.tensor([label])
            label = label.type(T.LongTensor)
            prediction = self.forward(input)
            loss = self.loss(prediction, label)
            prediction = F.softmax(prediction, dim=1)
            class_ = T.argmax(prediction, dim=1)
            wrong = T.where(class_ != label, 
                            T.tensor([1.]), 
                            T.tensor([0.]))
            
            acc = 1 - T.sum(wrong) / self.batch_size

            ep_acc.append(acc.item())
            ep_loss += loss.item()

        print('total loss %.3f' % ep_loss,
                'accuracy %.3f' % np.mean(ep_acc))