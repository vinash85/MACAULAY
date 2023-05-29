from functions import *
import torch
import torch.nn as nn
import umap.umap_ as umap
import seaborn as sns
import matplotlib.patches as mpl
import matplotlib.cm as cm


filename = 'datas/OmicsExpressionProteinCodingGenesTPMLogp1.csv'


df = pd.read_csv(filename)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
df.to_csv('datas/train_omics.csv', index=False)
df.to_csv('datas/test_omics.csv', index=False)


numeric_data_train = train_df.iloc[0:, 1:].values
numeric_data_test = test_df.iloc[0:, 1:].values
labels_train = train_df.iloc[:, 0].values
labels_test = test_df.iloc[:, 0].values
# Scale the data using StandardScaler
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(numeric_data_train)
scaled_data_test = scaler.transform(numeric_data_test)
# Split the data into train and test sets
# train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)
# Convert the data to PyTorch tensors
train_tensor = torch.tensor(scaled_data_train, dtype=torch.float32)
test_tensor = torch.tensor(scaled_data_test, dtype=torch.float32)
# Print the shapes of the data
print("df shape:", df.shape)
print('training tensor data shape:', train_tensor.shape)
print('test tensor data shape:', test_tensor.shape)
# Return the processed data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Define hyperparameters
input_dim = train_tensor.shape[1]  # Dimensionality of the omic gene expression data
learning_rate = 0.0001
batch_size = 128
num_epochs = 50
hidden_dim = 600
latent_dim = hidden_dim - 100
vae = VAE(input_dim, latent_dim, hidden_dim).to(device)

# Define reconstruction loss (e.g., Mean Squared Error)
reconstruction_loss = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Create DataLoader for batch training
dataset = TensorDataset(train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    running_recon_loss = 0.0


    for batch_data in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Move batch data to the appropriate device
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
    # Reconstruction example
# Pass the test data through the trained VAE model
torch.save(vae.state_dict(), f"vae_state/vae_expression.pth")
print (f'done')