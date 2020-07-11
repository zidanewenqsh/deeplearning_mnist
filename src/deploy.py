import torch
from net import NetMNIST

if __name__ == '__main__':
    param_path = r"D:\PycharmProjects\deeplearning_mnist\saves\mnistdemo\netmnist_00_0\netmnist_00_0.pt"
    script_path = r"D:\PycharmProjects\deeplearning_mnist\saves\mnistdemo\netmnist_00_0\netmnist_00_0.script"
    module = NetMNIST()
    module.load_state_dict(torch.load(param_path))
    module_script = torch.jit.script(module)
    module_script.save(script_path)
