from Generator import Generator
from Expert import Expert
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable

class GAN:

    def __init__(self, cuda: str, data_size: int, batch: int = 20):
        self.image_size = data_size
        self.noise_size = 100
        self.generator: Generator = Generator(self.noise_size, data_size, cuda)
        self.expert: Expert = Expert(data_size, cuda)
        self.GPU = cuda
        self.batch_size = batch
    
    def generate_noise(self) -> list[int]:
        return Variable(torch.randn((self.batch_size, self.noise_size))).to(self.GPU)

    def generate(self):
        noise = self.generate_noise()
        return self.generator(noise)
    
    def train_expert(self, batch, loss, adam):
        self.expert.zero_grad()
        
        images = batch.view(-1, self.image_size).to(self.GPU)
        real = Variable(torch.ones(self.batch_size, 1).to(self.GPU))
        experts_output_real = self.expert(images)
        real_loss = loss(experts_output_real, real)

        noise: List[int] = self.generate_noise()
        forged_image = self.generator(noise)
        forged = Variable(torch.zeros(self.batch_size, 1).to(self.GPU))

        experts_output_forged = self.expert(forged_image)
        forged_loss = loss(experts_output_forged, forged)

        experts_loss = (real_loss + forged_loss)

        experts_loss.backward()
        adam.step()

        return experts_loss.data.item()

    def train_forger(self, loss, adam):
        self.generator.zero_grad()
        noise: List[int] = self.generate_noise()
        real = Variable(torch.ones(self.batch_size, 1).to(self.GPU))

        forged_images = self.generator(noise)
        experts_output_forged = self.expert(forged_images)
        generators_loss = loss(experts_output_forged, real)
        generators_loss.backward()
        adam.step()

        return generators_loss.data.item()



    def train(self, data: DataLoader, epochs: int = 100) -> None:

        loss = torch.nn.BCELoss()
        learning_rate = 0.0002
        g_learn = optim.Adam(self.generator.parameters(), lr=learning_rate)
        e_learn = optim.Adam(self.expert.parameters(), lr=learning_rate)

        G_loss, E_loss = [], []

        for epoch in range(epochs):

            for i, (batch, _) in enumerate(data):

                experts_loss = self.train_expert(batch, loss, e_learn)
                generators_loss = self.train_forger(loss, g_learn)
                G_loss.append(generators_loss)
                E_loss.append(experts_loss)


            with torch.no_grad():
                noise = self.generate_noise()
                generated_image = self.generator(noise)
                generated_image = torch.reshape(generated_image, (-1, 1, 28, 28))
                save_image(generated_image, f"images/img_{epoch}.png")

            print(f"{epoch+1}: \n Expert: {torch.mean(torch.FloatTensor(E_loss)):.4f} \n Generator: {torch.mean(torch.FloatTensor(G_loss)):.4f}\n")
