from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import trainer
import model
import torch
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler


def main():
    train_data = datasets.FashionMNIST(
        root="/dev/shm/data",
        train=True,
        download=True,
        transform=ToTensor(), )
    train_data = DataLoader(train_data,
                            batch_size=500,
                            sampler=DistributedSampler(train_data))

    test_data = datasets.FashionMNIST(
        root="/dev/shm/data",
        train=False,
        download=True,
        transform=ToTensor(), )
    test_data = DataLoader(test_data,
                           batch_size=500,
                           sampler=DistributedSampler(test_data))

    print("Files loaded! (Fashion MNIST)", flush=True)
    train = trainer.Trainer(model.Model(), train_data, test_data, max_epochs=200, checkpoint_at=10)

    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    train.train()
    destroy_process_group()


if __name__ == "__main__":
    main()
