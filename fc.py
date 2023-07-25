import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

############ Neural Network ############################

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128) 
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)  
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x) 
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)  
        x = self.relu3(x)
        x = self.dropout(x)  
        logits = self.fc4(x)
        return logits



############### Load Both cell line embeddings and gene amino acid sequence embedings #################################

data1 = pd.read_csv("datas/final_cell_line_embedding.csv")
data2 = pd.read_csv("datas/final_gene_embeddings.csv")

# batch size of data to be concatenated and fed to the neural network 
batch_size1 = 500

# Create an empty list to store the batches of combined data
combined_batches = []

print('data read successfully')

############################## Loop to concatenate the two datasets in all possible forms ##################################
# Divide data1 into batches
for i in range(0, len(data1), batch_size1):
    batch_data1 = data1.iloc[i:i+batch_size1]

    # Divide data2 into batches separately
    for j in range(0, len(data2), batch_size1):
        combined_batches = []
        batch_data2 = data2.iloc[j:j+batch_size1]

        # Add a new column 'key' with a constant value for both DataFrames to perform a cross merge
        batch_data1['key'] = 1
        batch_data2['key'] = 1

        # Reorder the columns in the desired order for both DataFrames
        batch_data1 = batch_data1[list(data1.columns[1:]) + ['key']]
        batch_data2 = batch_data2[list(data2.columns[1:]) + ['key']]

        # Merge the data based on the 'key' column and drop the 'key' column afterward
        combined_batch = pd.merge(batch_data1, batch_data2, on='key').drop(columns=['key'])

        # Append the combined batch to the list of combined batches
        
        combined_batches.append(combined_batch)

        # Concatenate all batches to create the final merged DataFrame
        X_train = pd.concat(combined_batches)

        print(f'{i+batch_size1} of data1 is done')
        
##################### The first 500 Batches which forms 250000 rows after concat, are preprocessed and fed to the neural network under the same loop #####################
        
        # Convert training data to tensors
        X_train1 = torch.tensor(X_train.values, dtype=torch.float32)
        Y_train1 = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32)  # Reshape Y_train to (num_samples, 1)

        # Create a TensorDataset from training data
        train_data = TensorDataset(X_train1, Y_train1)

        # Create a DataLoader for training data
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

        # Convert test data to tensors
        X_test1 = torch.tensor(X_test.values, dtype=torch.float32)
        Y_test1 = torch.tensor(Y_test.values.reshape(-1, 1), dtype=torch.float32)  # Reshape Y_test to (num_samples, 1)

        # Create a TensorDataset from test data
        test_data = TensorDataset(X_test1, Y_test1)

        # Create a DataLoader for test data
        test_dataloader = DataLoader(test_data, batch_size=64)

        model = NeuralNetwork(input_dim)#.to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


        input_dim = X_train1.shape[1]
        num_epochs = 400

        ################# if greater than or equal to 500, Load previous model to continue training:
        if i >= 500:
            model.load_state_dict(torch.load('model_state/fc_model_state.pth'))

        print('model loaded successfully')

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss = 0.0
            model.train()
            for batch, (inputs, targets) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # if batch % 100 == 0:
                #     print(f"Batch {batch}/{len(train_dataloader)}, Loss: {loss.item()}")
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss}")
            
            model.eval()
            test_loss = 0.0

            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs = model(inputs)
                    test_loss += loss_fn(outputs, targets).item()

            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Average Test Loss: {avg_test_loss}")
            print("--------------------------------------------------")
        torch.save(model.state_dict(), 'model_state/fc_model_state.pth')
        print('model saved successfully')
