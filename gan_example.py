# generative-ai-models/gan_example.py

import torch
import torch.nn as nn
import torch.optim as optim

# Simple GAN example
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 784)

    def forward(self, z):
        return torch.sigmoid(self.fc(z))

# Example code to train the GAN model
if __name__ == "__main__":
    z = torch.randn(64, 100)  # Random noise vector
    gen = Generator()
    output = gen(z)
    print(output.shape)  # Should print torch.Size([64, 784])
