# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import timm
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def getHandwritingCharacterDefinitions():
    """
    Returns a dictionary with entries that define the names of each character, its length, and where the pen tip begins.
    
    Returns:
        charDef (dict)
    """
        
    charDef = {}
    
    #Define the list of all 31 characters and their names.
    charDef['charList'] = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                'greaterThan','comma','apostrophe','tilde','questionMark']
    charDef['charListAbbr'] = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                '>',',',"'",'~','?']

    #Define the length of each character (in # of 10 ms bins) to use for each template.
    #These were hand-defined based on visual inspection of the reconstructed pen trajectories.
    charDef['charLen'] = np.array([99, 91, 70, 104, 98, 125, 110, 104, 79, 92, 127, 68, 132, 90, 
                        84, 113, 104, 74, 86, 110, 86, 83, 110, 103, 115, 100, 82, 77, 116, 71, 110]).astype(np.int32)
    
    #For each character, this defines the starting location of the pen tip (0 = bottom of the line, 1 = top)
    charDef['penStart'] = [0.25, 1, 0.5, 0.5, 0.25, 1.0, 0.25, 1.0, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 1, 
           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 1, 0.5, 1]
    
    #dictionary to convert string representation to character index
    charDef['strToCharIdx'] = {}
    for x in range(len(charDef['charListAbbr'])):
        charDef['strToCharIdx'][charDef['charListAbbr'][x]] = x
        
    #ordering of characters that kaldi (i.e., the language model) expects
    charDef['kaldiOrder'] = ['<ctc>','>',"'",',','.','?','a','b','c','d','e','f','g','h','i','j',
                             'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    #re-indexing to match kaldi order (e.g., outputs[:,:,charDef['idxToKaldi']] places the output in kald-order)
    charDef['idxToKaldi'] = np.array([31,26,28,27,29,30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                     21,22,23,24,25]).astype(np.int32)
    
    return charDef

# Load the Dataset
topDirs = ['Datasets']
dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
            't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']
charDef = getHandwritingCharacterDefinitions()

all_tensors = []
all_labels = []
for directory in dataDirs:
    
    mat = f'./handwritingBCIData/Datasets/{directory}/singleLetters.mat'
    data = sio.loadmat(mat)
    ctr = 0
    for letter in charDef['charList']:
        t = torch.Tensor(data[f'neuralActivityCube_{letter}'])
        qty = t.shape[0]
        labels = torch.Tensor([ctr]*qty)
        ctr += 1
        all_tensors.append(t)
        all_labels.append(labels)

tensor_data = torch.cat(all_tensors, dim=0)
tensor_data = np.repeat(tensor_data[..., np.newaxis], 3, -1).transpose(-1,-2).transpose(-2,-3)
tensor_labels = torch.cat(all_labels).long()

# Data Preprocessing
dataset = TensorDataset(tensor_data, tensor_labels)
train_data, test_data = random_split(dataset, [3000, 627])
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# The ViT model
class ViT(nn.Module):
    def __init__(self, num_classes, patch_size=16, hidden_dim=768, num_heads=12, num_layers=6):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        self.embedding_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Patch Embedding
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.rand(1, 1 + (tensor_data.shape[-2] // patch_size), 1 + (tensor_data.shape[-3] // patch_size), hidden_dim))

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Classifier Head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)

        # Flatten and add positional encoding
        x = x.flatten(2).transpose(1, 2)
        x += self.positional_embedding

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classifier Head
        x = self.head(x)

        return x

# Instantiate the Model
num_classes = len(charDef['charList'])
model = ViT(num_classes)

# Model Compilation
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)

num_epochs = 100
batch_size = 32

# Training Loop
for epoch in range(num_epochs):
    model.train()
    print(f'epoch {epoch}')
    for batch in train_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels= labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)  # Add a channel dimension to the input
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()
    
    # Model Evaluation
    model.eval()
    with torch.no_grad():
        cumulative_accuracy = torch.tensor([]).to(device)
        for batch in test_dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            val_outputs = model(inputs).to(device)
            val_loss = criterion(val_outputs, labels).to(device)
            val_predictions = torch.argmax(val_outputs, dim=1).to(device)
            val_accuracy = (val_predictions == labels).float().to(device)
            cumulative_accuracy = torch.cat([cumulative_accuracy,val_accuracy], dim=0).to(device)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {cumulative_accuracy.mean().item():.4f}")

# Save Model
torch.save(model.state_dict(), './vit_single_chars_vit_1.pth')
model.load_state_dict(torch.load('./vit_single_chars_vit_1.pth'))

# Confusion Matrix
y_pred = []
y_true = []

# Iterate over test data
for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels= labels.to(device)
        # Resize...
        # inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)  # Resize inputs
        
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# Constant for classes
classes = charDef['charList']
classes = classes[:-5] + ['>', ',', '\'', '~', '?']
print(len(classes))

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (24//1.3,14//1.3))
sn.heatmap(df_cm, annot=True)
plt.savefig('output_vit_single_chars_1.png')
