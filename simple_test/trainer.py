import os.path

import torch.optim as optim
import torch.nn as neural
import torch.utils.data


class Trainer:
    def __init__(self,
                 model: neural.Module, train_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader,
                 optimiser: optim.Optimizer = None, max_epochs: int = 1,
                 loss_fn=neural.CrossEntropyLoss(), checkpoint_at: int = 1):

        self.model = model

        if optimiser is None:
            optimiser = optim.Adam(model.parameters())

        self.optimiser = optimiser
        self.max_epochs = max_epochs
        self.epochs = 0
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.checkpoint_every = checkpoint_at

        if os.path.isfile("simple.mlsv"):
            self._load()

        self.in_device = self.model._arch.devices[0]
        self.out_device = self.model._arch.devices[-1]

    def _save(self):
        state = {'model': self.model.state_dict(), 'epochs_run': self.epochs}
        torch.save(state, "simple.mlsv")
        print(f"Saving at Epoch {self.epochs + 1}")

    def _load(self):
        save = torch.load("simple.mlsv")
        self.model.load_state_dict(save['model'])
        self.epochs = save['epochs_run']

    def _train_batch(self, data, labels):
        self.optimiser.zero_grad()
        out = self.model(data)
        loss = self.loss_fn(out, labels)
        loss.backward()
        self.optimiser.step()

    def _train_epoch(self):
        # self.train_data.sampler.set_epoch(epoch)
        for data, labels in self.train_data:
            data = data.to(self.in_device)
            labels = labels.to(self.out_device)
            self._train_batch(data, labels)
            # print("One more batch done")

    def _test(self):
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_data:
                X, y = X.to(self.in_device), y.to(self.out_device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Epoch {self.epochs + 1} | Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>0.3f}", flush=True)

    def train(self):
        resume: int = self.epochs
        for ep in range(resume, self.max_epochs):
            self.epochs = ep
            self._train_epoch()
            self._test()
            if (ep + 1) % self.checkpoint_every == 0:
                self._save()
