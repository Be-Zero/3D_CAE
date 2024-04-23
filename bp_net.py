# coding=utf-8
import os
import numpy as np
import torch
from sklearn import metrics
from torch import nn


class BpNet(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, base_net, p=0.5):
        super(BpNet, self).__init__()
        net = []
        for i in range(len(base_net)-1):
            net.append(nn.Dropout(p=p))
            net.append(nn.Linear(base_net[i], base_net[i+1]))
            if i != len(base_net)-2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)
        self.to(self.device)

    def forward(self, x):
        x = self.net(x)
        return x

    def fit(self, train_loader, valid_loader, num_epochs=200, lr=0.001, loss_type='cross-entropy', optim_type='SGD'):
        # initialize
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight.data, gain=1)
        # nn.init.kaiming_uniform_(m.weight.data, a=0, nonlinearity='relu')

        # weight
        weight = torch.from_numpy(np.array([0.9955117572446093, 0.8606693335935213, 0.9190164894136014, 0.9768757927602693, 0.9528734510683969, 0.9287735388818421, 0.9972680261488925, 0.9533613035418089,
                                  0.9980485901063518, 0.9051614791686994, 0.7604644355546882, 0.942140696653332, 0.9799980485901063, 0.8765733242267538, 0.9623377890525905, 0.990925943994536])).float().to(self.device)
        if loss_type == 'cross-entropy':
            criterion = nn.CrossEntropyLoss(weight=weight)

        if optim_type == 'SGD':
            optim = torch.optim.SGD(params=self.parameters(), lr=lr)
        elif optim_type == 'Adam':
            optim = torch.optim.Adam(
                params=self.parameters(), lr=lr, betas=(0.9, 0.999))

        # validate
        self.eval()
        total_loss = 0.0
        total_num = 0
        valid_predict_lables = []
        valid_true_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.data * len(inputs)
                total_num += inputs.size()[0]
                for tl in labels:
                    valid_true_labels.append(tl.data.cpu())
                for pl in outputs:
                    valid_predict_lables.append(
                        torch.argmax(pl).data.cpu().type(torch.int8))

        valid_predict_lables = torch.tensor(valid_predict_lables)
        valid_true_labels = torch.tensor(valid_true_labels)
        max_valid_kappa = 0.0
        valid_loss = total_loss / total_num
        valid_acc = metrics.accuracy_score(
            valid_true_labels, valid_predict_lables)
        valid_kappa = metrics.cohen_kappa_score(
            valid_true_labels, valid_predict_lables)

        information = []
        print("#Epoch 0: Valid Loss: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}".format(
            valid_loss, valid_acc, valid_kappa))
        information.append(
            "#Epoch 0: Valid Loss: {:.4f}, acc: {:.4f}, Kappa: {:.4f}\n".format(valid_loss, valid_acc, valid_kappa))
        best_epoch = 0

        for epoch in range(num_epochs):
            # self.train()
            train_loss = 0.0
            train_predict_lables = []
            train_true_labels = []
            with torch.enable_grad():
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    train_loss += loss.data * len(inputs)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    for tl in labels:
                        train_true_labels.append(tl.data.cpu())
                    for pl in outputs:
                        train_predict_lables.append(
                            torch.argmax(pl).data.cpu().type(torch.int8))

            valid_loss = 0.0
            valid_predict_lables = []
            valid_true_labels = []
            # self.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.data * len(inputs)
                    for tl in labels:
                        valid_true_labels.append(tl.data.cpu())
                    for pl in outputs:
                        valid_predict_lables.append(
                            torch.argmax(pl).data.cpu().type(torch.int8))

            train_predict_lables = torch.tensor(train_predict_lables)
            train_true_labels = torch.tensor(train_true_labels)
            valid_predict_lables = torch.tensor(valid_predict_lables)
            valid_true_labels = torch.tensor(valid_true_labels)
            train_acc = metrics.accuracy_score(
                train_true_labels, train_predict_lables)
            valid_acc = metrics.accuracy_score(
                valid_true_labels, valid_predict_lables)
            train_kappa = metrics.cohen_kappa_score(
                train_true_labels, train_predict_lables)
            valid_kappa = metrics.cohen_kappa_score(
                valid_true_labels, valid_predict_lables)

            print("#Epoch {:3d}: Train Loss: {:.4f}, Valid Reconstruct Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Train Kappa: {:.4f}, Valid Kappa: {:.4f}".format(
                epoch+1, train_loss / len(train_loader.dataset), valid_loss / len(valid_loader.dataset), train_acc, valid_acc, train_kappa, valid_kappa))
            information.append("#Epoch {:3d}: Train Loss: {:.4f}, Valid Reconstruct Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Train Kappa: {:.4f}, Valid Kappa: {:.4f}\n".format(
                epoch+1, train_loss / len(train_loader.dataset), valid_loss / len(valid_loader.dataset), train_acc, valid_acc, train_kappa, valid_kappa))

            if max_valid_kappa <= valid_kappa:
                max_valid_kappa = valid_kappa
                best_epoch = epoch + 1
                self.save_model('./model/Indian_pines/bp_current.pt')
        rt_name = 'bp_ep{}_v{:.4f}'.format(best_epoch, max_valid_kappa)
        os.rename('./model/Indian_pines/bp_current.pt', './model/Indian_pines/' + rt_name + '.pt')
        with open('./logs/Indian_pines/' + rt_name + '.log', 'w') as f:
            for str in information:
                f.write(str)
        return rt_name

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
