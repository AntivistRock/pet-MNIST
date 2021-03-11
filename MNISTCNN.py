import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
from torchvision import datasets, transforms
from sklearn.metrics import f1_score


class CNN:
    def __init__(self, mode, device, path):
        self.device = torch.device(device)
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # bs_8_14_14
            nn.Conv2d(8, 16, (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # bs_16_7_7
            nn.Conv2d(16, 32, (3, 3), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # bs_32_3_3
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # bs_64_1_1
            nn.Conv2d(64, 10, (1, 1)),
            # nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        ).to(device)
        self.infer_transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        if mode == 'cold':
            self.valloader = self.get_loader(False, 64)
            self.trainloader = self.get_loader(True, 64)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
            self.criterion = nn.NLLLoss().to(self.device)
        else:
            self.set_weights('upload', path)
            print('Model is ready to predict.')

    @staticmethod
    def get_loader(train, batch_size):
        mnist = datasets.MNIST('mnist', train=train, download=True, transform=transforms.ToTensor())
        loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
        return loader

    def train(self):
        self.model.train()
        for X, y in self.trainloader:
            X = X.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(X).to('cpu').squeeze()
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

    def eval_f1(self):
        preds = []
        labels = []
        self.model.eval()
        for X, y in self.valloader:
            X = X.to('cuda')
            res = self.model(X).to('cpu').squeeze()
            res = res.argmax(dim=1)
            preds += res.detach().tolist()
            labels += y.detach().tolist()
        return f1_score(labels, preds, average='macro')

    def predict(self, X):
        self.model.eval()
        X = self.infer_transforms(X).to('cuda')
        pred = self.model(X.unsqueeze(0))
        number = pred.argmax()
        return number

    def set_weights(self, mode, path):
        if mode == 'save':
            torch.save(self.model.state_dict(), path)
            print('Weights saved.')
        elif mode == 'upload':
            self.model.load_state_dict(torch.load(*glob.glob(path + r'/*.pth')))
            print('Weights uploaded.')
        else:
            print('No such mode.')
