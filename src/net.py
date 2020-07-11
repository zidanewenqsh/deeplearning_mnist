import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        base = 32
        self.layer = nn.Sequential(
            # nn.Conv2d(1,base,3,1,1),
            # nn.ReLU(),
            # nn.Conv2d(base,base,3,1,1),
            # nn.ReLU(),
            nn.Conv2d(1, base, 3, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(base * 4, 10, 3, 1, 0),
        )
        self.layer2 = nn.Softmax(dim=-1)
        self.paraminit()
    def paraminit(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)

    def forward(self, x):
        y = self.layer(x)
        y = y.reshape(-1, 10)
        y = self.layer2(y)
        return y

if __name__ == '__main__':
    print(__name__)
    net = NetMNIST()

    exit()
    x = torch.randn(1,1,28,28)
    y = net(x)
    print(y.size())
    print(y)
    params = []
    for param in net.parameters():
        # print(param)
        params.extend(param.view(-1).cpu().detach().numpy())
    # print(params)
    # np.histogram(params,10)
    plt.hist(params,10000,(-0.5,0.5))
    plt.show()
    # module_script = torch.jit.script(net)
    # module_script.save("mnistmodule.pt")
    # torch.save(net,"net.pth")
    #net1 = torch.load("net.pth")
    # for n, p in net1.named_parameters():
    #     print(n)
    #     print(p)