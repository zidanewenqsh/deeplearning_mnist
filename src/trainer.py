import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from src.net import NetMNIST as Net
from torchvision import datasets, transforms
from base.baseai import BaseAI


class Trainer(BaseAI):
    def __init__(self, netName: str = None, cfgfile=r".\cfg.ini", net=Net):
        super(Trainer, self).__init__(netName=netName, cfgfile=cfgfile, net=net)
        print("initial complete")

    def _datasetinit(self):
        trainDataset = datasets.MNIST(
            root=r".\datas",
            train=True,
            transform=transforms.ToTensor(),
            download=False
        )
        self.trainDataloader = data.DataLoader(trainDataset, batch_size=self.batchSize, shuffle=True, drop_last=True,
                                               num_workers=self.numWorkers)
        testDataset = datasets.MNIST(
            root=r".\datas",
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )
        self.testDataloader = data.DataLoader(testDataset, batch_size=len(testDataset) // 100, shuffle=True,
                                              drop_last=True,
                                              num_workers=self.numWorkers)
        print("datasetlen", len(trainDataset))

    @property
    def _lossFn(self):
        return nn.MSELoss()

    @property
    def _optimizer(self):
        return optim.Adam(self.net.parameters())

    def get_accuracy(self, output, label, isTest=False):
        output_ = output.detach()
        output_ = torch.argmax(output_, dim=-1)
        accuracy = torch.mean(output_.eq(label.view(-1)).float())
        if isTest:
            print(output_.reshape(-1))
            print(label.reshape(-1))
        return accuracy

    def test(self):
        with torch.no_grad():
            self.net.eval()
            for img_t, label_t in self.testDataloader:
                output_t, label_t, _ = self.get_output(img_t, label_t)
                accuracy_t = self.get_accuracy(output_t, label_t, isTest=True)

                print(f"test accuracy: {accuracy_t:.4f}")
                self._log(isTest=True, accuracy=accuracy_t.item())
                break

    def get_output(self, img, label):
        img = img.to(self.device)
        label_onehot = self.onehot(label, cls=10).to(self.device)
        label = label.to(self.device)
        output = self.net(img)
        return output, label, label_onehot

    def train(self):
        self._datasetinit()
        self._loginit()
        dataloaderLen = len(self.trainDataloader)

        i = 0
        j = 0

        if len(self.logDict["i"]) > 0:
            i = self.logDict["i"][-1]
            j = self.logDict["j"][-1] + 1
            if j >= dataloaderLen:
                i += 1
                j = 0

        while (i < self.epoch):
            self.net.train()
            for img, label in self.trainDataloader:

                output, label, label_onehot = self.get_output(img, label)

                loss = self._lossFn(output, label_onehot)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if 0 == (j + 1) % self.printPoint or j == dataloaderLen - 1:
                    accuracy = self.get_accuracy(output, label)

                    print(f"epoch: {i:>4d}, batch: {j:>4d}, loss: {loss.item():.4f}, accuracy: {accuracy.item():.4f}")
                    torch.save(self.net.state_dict(), self.netfile)
                    self._log(i=i, j=j, loss=loss.item(), accuracy=accuracy.item())
                j += 1
                if j == dataloaderLen:
                    j = 0
                    break

            self.plot("loss", "accuracy")
            self.test()

            torch.save(self.net.state_dict(), self.netfile_backup)

            i += 1



if __name__ == '__main__':
    # baseai = BaseAI("onet_00_0",'./cfg.ini')
    trainer = Trainer("net_00_2", './cfg.ini')
    trainer.train()
    # imgdir = r"..\datas\train1"
    # imgtestdir = r"..\datas\test1"
    # dataset = Dataset(imgdir)
    # datasettest = Dataset(imgtestdir)

    # print(len(dataset))
