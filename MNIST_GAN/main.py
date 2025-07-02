import torch
from torchvision import datasets, transforms
from GAN import GAN
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(batch: int):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x * 2 - 1)                              
                                  ])
    train = datasets.MNIST(root='data', train=True, download=True, transform = transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform = transform)

    train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

    return train_loader, test_loader


def main():
    batch = 64
    train, test = get_data(batch)
    gan: GAN = GAN(device, 28*28, batch)
    gan.train(train, 100)


if __name__ == "__main__":
    main()

