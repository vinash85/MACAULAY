import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functions import *

# Load the data and preprocess it

train_df = pd.read_csv('datas/train_omics.csv')
numeric_data_train = train_df.iloc[:, 1:].values
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(numeric_data_train)
train_tensor = torch.tensor(scaled_data_train, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define hyperparameters
input_dim = train_tensor.shape[1]
learning_rate = 0.0001
batch_size = 128
num_epochs = 5000
hidden_dim = 600
latent_dim = hidden_dim - 100

# Define reconstruction loss
reconstruction_loss = nn.MSELoss()

# Create an instance of the VAE
vae = VAE(input_dim, latent_dim, hidden_dim)

# Load the saved model weights
vae.load_state_dict(torch.load('vae_state/vae_expression.pth'))

# Define the optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Define reconstruction loss (e.g., Mean Squared Error)
reconstruction_loss = nn.MSELoss()

# Create DataLoader for batch training
dataset = TensorDataset(train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    running_recon_loss = 0.0
    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_data = batch_data[0].to(device)

        # Forward pass
        output, mu, logvar = vae(batch_data)

        # Compute reconstruction loss
        recon_loss = reconstruction_loss(output, batch_data)

        # Backward pass and optimization
        recon_loss.backward()
        optimizer.step()

        # Accumulate running reconstruction loss
        running_recon_loss += recon_loss.item()

    # Print epoch statistics
    epoch_recon_loss = running_recon_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Reconstruction Loss: {epoch_recon_loss:.4f}")


# Save the updated VAE model
torch.save(vae.state_dict(), 'vae_state/vae_expression.pth')












