import torch
from torchvision import datasets, transforms
from GAN import GAN
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(batch: int):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))
                                  ])
    train = datasets.MNIST(root='data', train=True, download=True, transform = transform)
    test = datasets.MNIST(root='data', train=False, download=True, transform = transform)

    train_loader = DataLoader(dataset=train, batch_size=batch, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch, shuffle=True)

    return train_loader, train


def main():
    batch = 100
    epochs = 50
    train, train_data = get_data(batch)
    image_size = train_data.data.size(1) * train_data.data.size(2)
    gan: GAN = GAN(device, image_size, batch)
    gan.train(train, epochs)


if __name__ == "__main__":
    main()

