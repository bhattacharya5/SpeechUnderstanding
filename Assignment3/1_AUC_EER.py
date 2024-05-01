import collections
import torch
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import os
from model import SSLModel 

# Function to load the dataset
def load_dataset(data_dir):
    fake_files = [os.path.join(data_dir, "Fake", f) for f in os.listdir(os.path.join(data_dir, "Fake"))]
    real_files = [os.path.join(data_dir, "Real", f) for f in os.listdir(os.path.join(data_dir, "Real"))]
    files = fake_files + real_files
    labels = [1] * len(fake_files) + [0] * len(real_files)  # 1 for fake, 0 for real
    return files, labels

# Function to preprocess audio
def preprocess_audio(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    # Preprocess the waveform if needed (e.g., normalization)
    return waveform, sample_rate

# Load the pretrained model
model_path = "Best_LA_model_for_DF.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
# Load the model
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path, map_location=device)

#print(model)
# Check if the model is an OrderedDict (indicating it was loaded as state_dict)
if isinstance(model, collections.OrderedDict):
    # Create an instance of the model class and load the state_dict
    model_class = Model()
    model_class.load_state_dict(model)
    model = model_class

model.eval()
'''
model = SSLModel(device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
#model.eval()


# Load the dataset
data_dir = "Dataset_Speech_Assignment"
files, labels = load_dataset(data_dir)

# Lists to store predictions and true labels
predictions = []
true_labels = []

# Make predictions on the dataset
for file, label in zip(files, labels):
    waveform, _ = preprocess_audio(file)
    with torch.no_grad():
        output = model(waveform.unsqueeze(0))  # Assuming the model expects a batch dimension
    prediction = torch.sigmoid(output).item()  # Assuming the model outputs logits
    predictions.append(prediction)
    true_labels.append(label)

# Calculate AUC
auc = roc_auc_score(true_labels, predictions)

# Calculate EER
fpr, tpr, thresholds = roc_curve(true_labels, predictions, pos_label=1)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print("AUC:", auc)
print("EER:", eer)
