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
        transform=ToTensor(), ), batch_size=13684)

    test_data = DataLoader(datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor(), ), batch_size=13684)

    print("Files loaded! (Fashion MNIST)", flush=True)
    train = trainer.Trainer(model.Model(), train_data, test_data, max_epochs=20, checkpoint_at=5)

    train.train()


if __name__ == "__main__":
    main()
