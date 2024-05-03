import os
import librosa
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from model import Model  # Adjust this import based on the actual model class

# Assuming process_Rawboost_feature is a function to process audio features
from data_utils_SSL import process_Rawboost_feature, pad

# Dataset loader function
def load_dataset(data_dir, numb_files):
    fake_files = [os.path.join(data_dir, "Fake", f) for f in os.listdir(os.path.join(data_dir, "Fake")) if f.endswith('.mp3')]
    real_files = [os.path.join(data_dir, "Real", f) for f in os.listdir(os.path.join(data_dir, "Real")) if f.endswith('.mp3')]
    
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

# Main function
def main():
    data_dir = "Dataset_Speech_Assignment"
    model_path = "best_model.pth"
    numb_files = 10
    files, labels = load_dataset(data_dir, numb_files)
    dataset = CustomDataset(files, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #checkpoint = torch.load(model_path, map_location=device)
    #print("Keys in checkpoint:", checkpoint.keys())

    model = load_pretrained_model(model_path, device)    
    
    auc, eer = evaluate_model(model, dataloader, device)
    print(f"AUC: {auc}, EER: {eer}")

if __name__ == "__main__":
    main()