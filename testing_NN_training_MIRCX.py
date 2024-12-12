#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:55:53 2024

@author: nebrahimkutt
"""
#%%

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import gc
import re
#%%

class IntensityDataset(Dataset):
    def __init__(self, data_dir, normalize=True):
        """
        Initialize the dataset by reading and processing all files in the directory.
        
        Args:
        - data_dir (str): Path to the directory containing the .txt files.
        - normalize (bool): Whether to normalize the input features and target.
        """
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir)]
        self.data = []
        self.normalize = normalize
        self._process_files()

    def _extract_teff_logg(self, file_name):
        """
        Extract Teff, logg, and z from the file name.
        Assumes the file name format: 't####_g#.##_z#.##...'
        """
        match = re.search(r"t(\d{4})_g(\d\.\d{2})_z([+-]?\d\.\d{2})", file_name)
        if match:
            teff = int(match.group(1))  # Teff as an integer
            logg = float(match.group(2))  # logg as a float
            z = float(match.group(3))  # z as a float
            return teff, logg, z
        else:
            raise ValueError("Filename format does not match the expected pattern")


    def _process_files(self):
        """
        Process all files in the directory to populate self.data with feature-target pairs.
        """
        for file_name in self.file_names:
            try:
                # Extract Teff and logg from the file name
                teff, logg, z = self._extract_teff_logg(file_name)
                file_path = os.path.join(self.data_dir, file_name)
                
                # Read the file content
                with open(file_path, "r") as f:
                    lines = f.readlines()
                
                # Extract Radau points
                radau_points = np.array(
                    [float(value) for value in lines[1].split("mu=(")[1].split(")")[0].split()]
                )
                
                # Parse intensities for each wavelength and Radau point
                for line in lines[2:]:
                    values = np.array(line.split(), dtype=np.float32)
                    wavelength = values[0]
                    intensities = values[1:]  # Intensity values for this wavelength
                    
                    for mu, intensity in zip(radau_points, intensities):
                        features = [teff, logg, wavelength, mu]
                        if self.normalize:
                            features = self._normalize_features(features)
                            intensity = self._normalize_target(intensity)
                        self.data.append((features, intensity))
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    def _normalize_features(self, features):
        """
        Normalize the features according to the specified scaling:
        - Teff: Divide by 1000
        - Wavelength: Divide by 1e-6
        - Radau: Divide by 1e-1
        """
        teff, logg, wavelength, radau = features
        teff /= 1000.0
        wavelength /= 1e-6
        radau /= 1e-1
        return [teff, logg, wavelength, radau]

    def _normalize_target(self, intensity):
        """
        Normalize the target intensity by dividing it by 1e6.
        """
        return intensity / 1e6

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        Args:
        - idx (int): Index of the sample.
        Returns:
        - features (torch.Tensor): Input features [Teff, logg, wavelength, 1 - Radau point].
        - target (torch.Tensor): Intensity value.
        """
        features, target = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    


class Save_Best_Model():
    """Class to save the best model while training. If the current epochs validation loss is
    less than the previous least less, then save the model state."""
    
    def __init__(self, best_valid_loss =float("inf")):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion,loss,loss_eval, path):
        
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            # torch.save(model.state_dict(),'test_figures/training_50A/best_model.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn': criterion,
                'loss':loss,
                'loss_eval':loss_eval
                }, path)


def save_model(epochs, model, optimizer, criterion,loss,loss_eval, path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    # torch.save(model.state_dict(),'test_figures/training_50A/final_model.pth')
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn': criterion,
                'loss':loss,
                'loss_eval':loss_eval
                }, path)
    


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

#%%

models = "/DATA/Workhome/nebrahimkutt/MARCS_data/grid_spectra_binned_59wavelength"
dataset = IntensityDataset(models, normalize=True)

path_best_model = "/DATA/Workhome/nebrahimkutt/Codes/Results/test_SPICA_dec_2024/best_model_SPICA_LR_59wlen_1int.pth"
path_final_model = "/DATA/Workhome/nebrahimkutt/Codes/Results/test_SPICA_dec_2024/final_model_SPICA_LR_59wlen_1int.pth"

#%%

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

# DataLoaders
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%%

lr =0.0001
epoch = 50

# Initialize the model, loss function, and optimizer
model = ANN()
save_best_model = Save_Best_Model()         ### save the best model
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=lr)

count = 0
ls = []
ls_eval = []
pred_val = []
target_val = []

# Training loop

while count < epoch:
    total_train_loss = 0
    total_eval_loss = 0
    model.train()
    
    for idx, batch in enumerate(train_loader):
        total_loss = 0
        features, target = batch
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        pred_val.append(output)
        target_val.append(target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_train_loss += total_loss
        
    ls.append(total_train_loss)
    count += 1
    
    del features, target, output, loss
    gc.collect()
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            valid_loss = 0
            features, target = batch
            output_val = model(features)
            loss_val = criterion(output_val, target)
            valid_loss += loss_val.item()
            total_eval_loss += loss_val.item()
        ls_eval.append(total_eval_loss)
        save_best_model(total_eval_loss, count, model, optimizer, criterion,ls,ls_eval, path_best_model)
    print(f"best training loss: {total_train_loss}")
    print(f"best validation loss: {total_eval_loss}")
    print(f"Epoch: {count+1}\n")
    
    del features, target, output_val, loss_val
    gc.collect()

save_model(count, model, optimizer, criterion,ls,ls_eval, path_final_model)

torch.save({'train_set': train_loader, 'val_set': val_loader, 'test_set': test_loader}, '/DATA/Workhome/nebrahimkutt/Codes/Results/test_SPICA_dec_2024/data_loaders_SPICA_LR_59wlen_1int.pth')

#%%

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ls, label='training loss')
plt.plot(ls_eval, label='evaluation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
# plt.savefig('test_figures/loss.png')
plt.show()

#%%