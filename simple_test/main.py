from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import trainer
import model


def main():
    train_data = DataLoader(datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(), ), batch_size=500)

    test_data = DataLoader(datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(), ), batch_size=10000)

    print("Files loaded! (Fashion MNIST)", flush=True)
    train = trainer.Trainer(model.Model(), train_data, test_data, max_epochs=200, checkpoint_at=10)

    train.train()


if __name__ == "__main__":
    main()
