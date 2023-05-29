from functions import *
import torch
import torch.nn as nn
import umap.umap_ as umap
import seaborn as sns
import matplotlib.patches as mpl
import matplotlib.cm as cm


prs = Presentation()


test_df = pd.read_csv('datas/test_omics.csv')
numeric_data_test = test_df.iloc[0:, 1:].values
labels_test = test_df.iloc[:, 0].values
scaler = StandardScaler()
scaled_data_test = scaler.fit_transform(numeric_data_test)
test_tensor = torch.tensor(scaled_data_test, dtype=torch.float32)
print('test tensor data shape:', test_tensor.shape)







# Define hyperparameters
input_dim = test_tensor.shape[1]  # Dimensionality of the omic gene expression data
learning_rate = 0.0001
batch_size = 128
num_epochs = 20
hidden_dim = 600
latent_dim = hidden_dim - 100


# Define reconstruction loss (e.g., Mean Squared Error)
reconstruction_loss = nn.MSELoss()

# Create an instance of the VAE
vae = VAE(input_dim, latent_dim, hidden_dim)

# Load the saved model weights
vae.load_state_dict(torch.load('vae_state/vae_expression.pth'))

# Set the VAE to evaluation mode
vae.eval()


# Forward pass on the new dataset

latent_space = []
with torch.no_grad():
    _, mu, _ = vae(test_tensor)
latent_space.append(mu.cpu().detach().numpy())

latent_space = np.concatenate(latent_space, axis=0)
umap_embedding = umap.UMAP().fit_transform(latent_space)

metadata = pd.read_csv("Model.csv")
merged_data = pd.merge(test_df, metadata, left_index=True, right_index=True, how="inner")
# column_cont = ['SourceType','Sex']
column_cont = ["CellLineName","StrippedCellLineName","SourceType","SangerModelID","RRID","DepmapModelType","GrowthPattern","MolecularSubtype","PrimaryOrMetastasis","SampleCollectionSite","Sex","SourceDetail","CCLEName","PublicComments","OncotreeCode","OncotreeSubtype","OncotreePrimaryDisease"]
for j, content in enumerate(column_cont):   
    metadata_subset = merged_data[[content]]
    # Get the values of the column we're interested in and create a colormap for them
    SSC_metadata_subset = merged_data[[content]]
    unique_categories = SSC_metadata_subset[content].unique()
    num_categories = len(unique_categories)
    cmap = cm.get_cmap('tab20', num_categories)


    category_to_color = {category: cmap(i) for i, category in enumerate(unique_categories)}
    colors_by_category = [category_to_color[category] for category in SSC_metadata_subset[content]]



    # Create a separate figure for the legend
    legend_fig = plt.figure()
    legend_ax = legend_fig.add_subplot(111)

    handles = [mpl.Patch(color=category_to_color[category], label=category) for category in unique_categories]
    legend_ax.legend(handles=handles, bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.)
    legend_ax.axis('off')  # Turn off the axis to only show the legend
    # Add a title to the legend figure
    legend_fig.suptitle(f'Legend of {content}')
    # Save the legend as a separate image
    legend_fig.savefig(f'plot_images/the_legend_{content}.png')
    plt.close(legend_fig)  # Close the legend figure

    # Plot the UMAP scatterplot
    sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], c=colors_by_category)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title(f'UMAP Visualization of Latent Space ({content})')

    # Save the scatterplot image
    plt.savefig(f'plot_images/the_{content}_umap.png')
    plt.close()  # Close the scatterplot figure

    


for j, content in enumerate(column_cont):   
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(f'plot_images/the_{content}_umap.png', Inches(1), Inches(1), width=Inches(8), height=Inches(6))
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(f'plot_images/the_legend_{content}.png', Inches(1), Inches(1), width=Inches(8), height=Inches(6))



prs.save(f'slides/Deep_Plot_Essentiality.pptx')

























