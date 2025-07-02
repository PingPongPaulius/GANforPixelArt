from Generator import Generator
from Expert import Expert
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchvision.utils import save_image

class GAN:

    def __init__(self, cuda: str, data_size: int, batch: int = 20):
        self.image_size = data_size
        self.noise_size = 64
        self.generator: Generator = Generator(self.noise_size, data_size, cuda)
        self.expert: Expert = Expert(data_size, cuda)
        self.GPU = cuda
        self.batch_size = batch
    
    def generate_noise(self) -> list[int]:
        return torch.rand((self.batch_size, self.noise_size)).to(self.GPU)

    def generate(self):
        noise = self.generate_noise()
        return self.generator(noise)

    def train(self, data: DataLoader, epochs: int = 100) -> None:

        loss = torch.nn.BCELoss()
        learning_rate = 2e-4
        betas = (0.9,0.999)
        g_learn = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)
        e_learn = optim.Adam(self.expert.parameters(), lr=learning_rate, betas=betas)

        forged = torch.zeros(self.batch_size).to(self.GPU)
        real = torch.ones(self.batch_size).to(self.GPU)

        fixed_noise = self.generate_noise()

        for epoch in range(epochs):

            for i, (batch, _) in enumerate(data):

                # Train Discriminator
                images = batch.view(-1, self.image_size).to(self.GPU)
                if (images.shape[0] != self.batch_size):
                    continue
                experts_output_real = self.expert(images).view(-1)
                self.expert.zero_grad()

                noise: List[int] = self.generate_noise()
                forged_image = self.generator(noise)
                experts_output_forged = self.expert(forged_image).view(-1)

                real_loss = loss(experts_output_real, real)
                forged_loss = loss(experts_output_forged, forged)

                experts_loss = (real_loss + forged_loss)/2
                experts_loss.backward()
                e_learn.step()

                #Train Generator
                self.generator.zero_grad()

                forged_images = self.generator(noise)
                generators_loss = loss(self.expert(forged_images).view(-1), real)
                generators_loss.backward()
                g_learn.step()


            with torch.no_grad():
                self.test(epoch, fixed_noise)

            print(f"{epoch+1}: \n Expert: {experts_loss.item():.4f} \n Generator: {generators_loss.item():.4f}\n")

    def test(self, epoch, fixed_noise):
        generated_image = self.generator(fixed_noise)
        generated_image = torch.reshape(generated_image, (-1, 1, 28, 28))
        save_image(generated_image, f"images/img_{epoch}.png")
