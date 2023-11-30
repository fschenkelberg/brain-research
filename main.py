# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from torchvision.models import vision_transformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import timm

from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments

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

# Model Compilation
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100

# The ViT model
pretrained_model = 'google/vit-base-patch16-224-in21k'
model = timm.create_model(pretrained_model, pretrained=True)
original_head = model.head
model.head = nn.Identity()
num_classes = len(charDef['charList'])
model.head = nn.Linear(original_head.in_features, num_classes)
model.to(device)

# Create the ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model)

# Instantiate the ViT model
vit_classifier = model.parameters()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            inputs = feature_extractor(images=inputs, return_tensors="pt").pixel_values
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluate function (unchanged)
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            inputs = feature_extractor(images=inputs, return_tensors="pt").pixel_values
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test accuracy: {accuracy * 100:.2f}%")

# Train and evaluate the model
train(model, train_dataloader, criterion, optimizer, num_epochs)
evaluate(model, test_dataloader)

# Save Model
torch.save(model.state_dict(), './vit_single_chars.pth')
model.load_state_dict(torch.load('./vit_single_chars.pth'))

# Confusion Matrix
y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels= labels.to(device)
        inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)  # Resize inputs
        
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = charDef['charList']
classes = classes[:-5] + ['>', ',', '\'', '~', '?']
print(len(classes))
# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (24//1.3,14//1.3))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
