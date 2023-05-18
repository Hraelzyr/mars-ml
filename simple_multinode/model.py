import torch.nn as neural


class Model(neural.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = neural.Flatten()
        self._arch = neural.Sequential(neural.Linear(28*28, 128), neural.ReLU(),
                                       neural.Linear(128, 128), neural.ReLU(),
                                       neural.Linear(128, 10), neural.ReLU())

    def forward(self, inp):
        return self._arch(self.flatten(inp))
