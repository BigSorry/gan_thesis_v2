import torch.nn as nn
import torch

class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dimension=100):
        super(Generator, self).__init__()
        self.latent = 100
        self.n_out = 2
        self.fc0 = nn.Sequential(
            nn.Linear(self.latent, 256),
            #nn.LeakyReLU(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            #nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            #nn.LeakyReLU(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, self.n_out),
        )

    # forward method
    def forward(self, input):
        x = self.fc0(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, gaussian_dimension=2):
        super(Discriminator, self).__init__()
        self.n_in = gaussian_dimension
        self.n_out = 1
        self.fc0 = nn.Sequential(
            nn.Linear(self.n_in, 1024),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, self.n_out),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, input):
        x = self.fc0(input)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)

