import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pptx import Presentation
from pptx.util import Inches
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap


def process_data(filename):
    # Import the data
    df = pd.read_csv(filename)
    # Extract the numeric values from the DataFrame
    numeric_data = df.iloc[0:, 1:].values
    labels = df.iloc[:, 0].values
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    # Split the data into train and test sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)
    # Convert the data to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    # Print the shapes of the data
    print("df shape:", df.shape)
    print('training tensor data shape:', train_tensor.shape)
    # Return the processed data
    return train_tensor, labels, numeric_data, scaled_data

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Encoded layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

        )

        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        output = self.decoder(z)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
    


def device(train_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = train_tensor.to(device)
    return train_tensor

def create_dataloader(train_tensor, batch_size):
    # Create a TensorDataset from the train_tensor
    dataset = TensorDataset(train_tensor)

    # Create a DataLoader for batch training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Return the dataloader
    return dataloader


def plot_correlation_graph(test_tensor, correlation_coefficients):
    plt.figure()
    plt.scatter(range(test_tensor.shape[1]), correlation_coefficients)
    plt.xlabel('Gene Index')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Correlation of Reconstructed Genes (Test Data)')
    # plt.show()
    plt.savefig(f'plot_images/Correlation of Reconstructed Genes (Test Data)')

def plot_correlation_histogram(correlation_coefficients):
    plt.figure()
    plt.hist(correlation_coefficients, bins=30, edgecolor='black')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Histogram of Correlation Coefficients')
    # plt.show()
    plt.savefig('plot_images/Histogram of Correlation Coefficients')