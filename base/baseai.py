import os
import abc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import matplotlib.pyplot as plt
import configparser

from src.utils import *


class BaseAI(metaclass=abc.ABCMeta):
    def __init__(self, netName, cfgfile, net):

        self.cfgfile = cfgfile

        parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self._argparser(parser)

        if self.args.name == None:
            if netName != None:
                self.netName = netName
            else:
                raise ValueError
        else:
            self.netName = self.args.name

        self._deviceinit()
        self._cfginit(cfgfile)
        self._moudleinit(net)

    def _deviceinit(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device0 = torch.device("cuda:0")
            self.device1 = torch.device("cuda:1")
            self.device2 = torch.device("cuda:2")
            self.device3 = torch.device("cuda:3")


    def _cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
        self.saveDir = config.get(self.netName, "SAVE_DIR")
        self.imgDir = config.get(self.netName, "IMG_DIR")
        self.imgTestDir = config.get(self.netName, "IMGTEST_DIR")
        self.labelDir = config.get(self.netName, "LABEL_DIR")

        self.epoch = self.args.epoch if self.args.epoch else config.getint(self.netName, "EPOCH")

        self.alpha = self.args.alpha if self.args.alpha else config.getfloat(self.netName, "ALPHA")

        self.batchSize = self.args.batchsize if self.args.batchsize else config.getint(self.netName,
                                                                                       "BATCHSIZE")
        self.numWorkers = self.args.numworkers if self.args.numworkers else config.getint(self.netName,
                                                                                          "NUMWORKERS")
        self.printPoint = self.args.printpoint if self.args.printpoint else config.getint(self.netName,
                                                                                          "PRINTPOINT")
        self.subSaveDir = os.path.join(self.saveDir, self.netName)
        makedir(self.subSaveDir)

    def _argparser(self, parser):
        parser.add_argument("-n", "--name", type=str, default=None, help="the netfile name to train")
        parser.add_argument("-e", "--epoch", type=int, default=None, help="number of epochs")
        parser.add_argument("-b", "--batchsize", type=int, default=None, help="mini-batch size")
        parser.add_argument("-w", "--numworkers", type=int, default=None,
                            help="number of threads used during batch generation")
        parser.add_argument("-l", "--lr", type=float, default=None, help="learning rate for gradient descent")
        parser.add_argument("-r", "--printpoint", type=int, default=None, help="print frequency")
        parser.add_argument("-t", "--testpoint", type=int, default=None,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float, default=None, help="ratio of conf and offset loss")
        return parser.parse_args()

    def _moudleinit(self, moudle):
        self.netfile = os.path.join(self.subSaveDir, f"{self.netName}.pt")
        self.netfile_backup = os.path.join(self.subSaveDir, f"{self.netName}_backup.pt")
        self.net = moudle().to(self.device)
        if os.path.exists(self.netfile):
            self.net.load_state_dict(torch.load(self.netfile))
            print("load successfully")

    def _log(self, isTest: bool = False, **kwargs):
        for key, value in kwargs.items():
            self.logDict[key].append(value)
        if isTest:
            torch.save(self.logDict, self.logFileTest)
            torch.save(self.logDict, self.logFileTest_backup)
            torch.save(self.logDict, self.logFile_backup)
        else:
            torch.save(self.logDict, self.logFile)

    def _loginit(self):
        self.logFile = os.path.join(self.subSaveDir, f"{self.netName}.log")
        self.logFileTest = os.path.join(self.subSaveDir, f"{self.netName}_test.log")
        self.logFile_backup = os.path.join(self.subSaveDir, f"{self.netName}_backup.log")
        self.logFileTest_backup = os.path.join(self.subSaveDir, f"{self.netName}_test_backup.log")
        if os.path.exists(self.logFile):
            self.logDict = torch.load(self.logFile)
        else:
            self.logDict = {"i": [], "j": [], "loss": [], "accuracy": []}

        if os.path.exists(self.logFileTest):
            self.logDictTest = torch.load(self.logFileTest)
        else:
            self.logDictTest = {"i": [], "j": [], "accuracy": []}

    def onehot(self, a, cls=2):
        b = torch.zeros(a.size(0), cls).scatter_(-1, a.view(-1, 1).long(), 1).to(self.device)
        return b

    def plot(self, *args):
        for item in args:
            plotName = f"plot_{item}.jpg"
            plotPath = os.path.join(self.subSaveDir, plotName)
            y = np.array(self.logDict[item])
            plt.clf()
            plt.title(item)
            plt.plot(y)
            plt.savefig(plotPath)

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _datasetinit(self):
        pass

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def detect(self, *args, **kwargs):
        raise NotImplementedError

    def analyze(self, *args, **kwargs):
        raise NotImplementedError
