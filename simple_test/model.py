import torch.nn as neural
from torchgpipe import GPipe

class Model(neural.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = neural.Flatten()
        self._arch = GPipe(neural.Sequential(neural.Linear(28*28, 128), neural.ReLU(),
                                             neural.Linear(128, 128), neural.ReLU(),
                                             neural.Dropout(0.5),
                                             neural.Linear(128, 10), neural.ReLU()), balance=[4, 4], chunks=8)

    def forward(self, inp):
        return self._arch(self.flatten(inp))
