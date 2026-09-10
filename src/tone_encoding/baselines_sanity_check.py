import glob
import os
import pickle
import re
from matplotlib.pyplot import savefig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import MT19937, RandomState, SeedSequence
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tone_encoding.classification import (custom_train_test_split,
                                          load_input_and_labels_and_mask)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(p=0.2))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(p=0.3))

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def MLP_training_pipeline(X_train, X_test, y_train, y_test,
                          hidden_size = 2000,
                          num_hidden_layers = 4,
                          batch_size = 1000,
                          lr = 0.005,
                          num_epochs = 60
                          ):


    trainloader = torch.utils.data.DataLoader(list(zip(X_train,y_train)), batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(list(zip(X_test,y_test)), batch_size=batch_size, shuffle=True, num_workers=1)

    input_size = X_train.shape[-1]
    output_size = len(np.unique(y_train))

    model = MLP(input_size, hidden_size, output_size, num_hidden_layers).to(device)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    # #ryant et al setup

    # Run the training loop
    for epoch in tqdm(range(0, num_epochs)):         
        tqdm.write(f'Starting epoch {epoch+1}')
        epoch_loss = 0.0
        # Iterate over the DataLoader for training data
        for i, (inputs, targets) in enumerate(trainloader, 0):
            # Get inputs
            inputs = torch.nan_to_num(inputs.type(torch.float32)).to(device)
            targets = torch.tensor([int(x) for x in targets]) - 1
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)
            # Compute loss
            loss = loss_function(outputs.cpu(), targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            epoch_loss += loss.item()
        average_epoch_loss = epoch_loss / len(trainloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}")
    # Set the model to evaluation mode
    model.eval()

    # Variables to keep track of validation loss and accuracy
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    # Validation step
    with torch.no_grad():
        for inputs, targets in testloader:
            # Forward pass
            inputs = torch.nan_to_num(inputs.type(torch.float32)).to(device)
            targets = torch.tensor([int(x) for x in targets]) -1
            outputs = model(inputs).cpu()
            # Compute the loss
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    # Calculate average loss and accuracy
    average_loss = total_loss / len(testloader)
    accuracy = correct_predictions / total_samples

    return {"avg_loss": average_loss,
            'accuracy': accuracy}

def baseline_sanity_check():
    emb_files = ['data/f0-ryant_thchs30_extracted-data.pt',
                 'data/mfcc-ryant_thchs30_extracted-data.pt',
                 'data/facebook-wav2vec2-base_thchs30_extracted-data.pt',
                #  'data/facebook-wav2vec2-base_thchs30_extracted-data_cnn.pt',
                 ]
    contrast = 'tone'
    modes = ['alldata', 'heldout']
    seed = 42
    results_path = 'results/sanity_check'
    save_file = os.path.join(results_path,'sanity_check_results.csv')
    results = pd.read_csv(save_file, index_col=0).to_dict(orient='records') if os.path.isfile(save_file) else []
    embs_with_existing_results = [x['emb_file'] for x in results]
    rs = RandomState(MT19937(SeedSequence(seed))) # setting the random state for the random choice generator
    for emb_file in emb_files:
        if os.path.basename(emb_file) in embs_with_existing_results:
            print(f'{emb_file} results are present already')
            continue
        for mode in modes:
            _, _, all_inputs_arr, all_labels_arr = torch.load(emb_file)
            X, y, mask_array = load_input_and_labels_and_mask(all_inputs_arr, all_labels_arr, rs, contrast = contrast, mode = mode)
            for layer in tqdm(range(X.shape[1]), desc='Layers'):
                X_ = X[:,layer,:]
                X_train, X_test, y_train, y_test = custom_train_test_split(X_, 
                                                        y, 
                                                        mask_array = mask_array, 
                                                        seed = seed)
                
                layerwise_result = MLP_training_pipeline(X_train, X_test, y_train, y_test,
                                                        hidden_size = 2000,
                                                        num_hidden_layers = 4,
                                                        batch_size = 10000,
                                                        lr = 5e-3,
                                                        num_epochs = 60
                                                        )
                results.append({'emb_file': os.path.basename(emb_file),
                                'layer': layer,
                                'mode': mode,} | layerwise_result)
    df = pd.DataFrame(results)
    df.to_csv(save_file)

def main():
    baseline_sanity_check()


if __name__ == "__main__":
    main()