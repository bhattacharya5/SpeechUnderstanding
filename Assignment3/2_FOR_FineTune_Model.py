import os
import librosa
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from model import Model  # Adjust this import based on the actual model class
from data_utils_SSL import process_Rawboost_feature, pad

# Dataset loader function
def load_dataset(data_dir, numb_files):
    fake_files = [os.path.join(data_dir, "fake", f) for f in os.listdir(os.path.join(data_dir, "fake")) if f.endswith('.wav')]
    real_files = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith('.wav')]
    
    # Randomly sample files if the number of available files is greater than numb_files
    if len(fake_files) > numb_files:
        fake_files = random.sample(fake_files, numb_files)
    if len(real_files) > numb_files:
        real_files = random.sample(real_files, numb_files)
    
    files = fake_files + real_files
    labels = [1] * numb_files + [0] * numb_files  # Corrected label assignment
    return files, labels

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        # Load and process your .mp3 file here
        audio, sr = librosa.load(file_path, sr=16000)
        # Process audio file (e.g., feature extraction)
        audio_processed = process_Rawboost_feature(audio, sr, args=None, algo=None)  # Adjust args and algo as needed
        audio_padded = pad(audio_processed)
        return torch.tensor(audio_padded, dtype=torch.float32), label

def load_pretrained_model(model_path, device):
    # Create a minimal mock args object
    class Args:
        pass  # Add any expected attributes here if needed

    args = Args()
    
    model = Model(args=args,device=device)

    # Load the pre-trained model state dictionary
    pretrained_dict = torch.load(model_path, map_location=device)
    
    # Filter out unnecessary keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # Update the current model's state dictionary with the filtered state dictionary
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    return model

# Function to evaluate the model and calculate AUC and EER
def evaluate_model(model, dataloader, device):
    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Assuming the model outputs logits for two classes
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_scores.extend(scores)
            y_true.extend(labels.numpy())
    
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return auc, eer

# Train model function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validate the model if AUC improves
        val_loss, _ = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            print("Best model saved")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

def main():
    training_data_dir = "for-2sec/for-2seconds/training"
    testing_data_dir = "for-2sec/for-2seconds/testing"
    validation_data_dir = "for-2sec/for-2seconds/validation"
    model_path = "Best_LA_model_for_DF.pth"
    numb_files = 5
    
    # Load datasets
    train_files, train_labels = load_dataset(training_data_dir, numb_files)
    val_files, val_labels = load_dataset(validation_data_dir, numb_files)
    test_files, test_labels = load_dataset(testing_data_dir, numb_files)
    print("load_dataset completed")
    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_files, train_labels)
    val_dataset = CustomDataset(val_files, val_labels)
    test_dataset = CustomDataset(test_files, test_labels)
    print("Custom dataset loaded")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("data Loader completed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained model
    model = load_pretrained_model(model_path, device)
    print("Pre Trained model loaded")
    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model Training Started.")
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    print("Model training completed.")
    # Evaluate the model on test data
    auc, eer = evaluate_model(model, test_loader, device)
    print(f"AUC on test set after fine-tuning: {auc}, EER on test set after fine-tuning: {eer}")

if __name__ == "__main__":
    main()
