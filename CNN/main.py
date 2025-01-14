# Generate images of food

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class Generative(nn.Module):
    def __init__(self, batch_size, latent_dim):
        super(Generative, self).__init__()
                
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.output = nn.ConvTranspose2d(64, 3, kernel_size=4)        
        self.tanh   = nn.Tanh()
        
    def forward(self, noise):
        # Fully connected layers
        x = self.fc1(noise)
        # Reshape for ConvTranspose2d
        x = x.view(-1, 256, 1, 1)
        # Deconvolution layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        # Output layer
        x = self.output(x)
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * (input_shape[1] // 8) * (input_shape[2] // 8), 1),
            nn.Sigmoid() 
        )

    
    def forward(self, x):
        return self.model(x)
