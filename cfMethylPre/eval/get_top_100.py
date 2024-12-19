import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import umap.umap_ as umap





device = torch.device("cuda")

test_label_file = 'BC.txt'
with open(test_label_file, 'r') as f:
    lines = f.readlines()
    test_data = [line.strip().split() for line in lines]
    test_filenames = [item[0] for item in test_data]
    test_labels = [int(item[1]) for item in test_data]

    
class MyDataset(Dataset):
    def __init__(self, filenames, labels):
        self.filenames = filenames
        self.labels = labels

        self.data = []
        for filename in tqdm(filenames, desc='Reading files'):
            x = torch.from_numpy(np.loadtxt(filename)).float()
            x = x.reshape(1, 6585, 128)
            self.data.append(x)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1256)

splits = list(kf.split(test_filenames, test_labels))

train_index = splits[0][0]

train_filenames = [test_filenames[i] for i in train_index]
train_labels = [test_labels[i] for i in train_index]

train_dataset = MyDataset(train_filenames, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

from torch.autograd import Variable
model = models.resnet101(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('BC_1.pt'))
model.to(device)

model.eval()

row_scores = {}

true_labels = []
pred_labels = []

total_gradients = torch.zeros(6585, 128).to(device)

for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Testing')):

    if labels.item() == 1:
        inputs = inputs.to(device)
        inputs = Variable(inputs, requires_grad=True)


        outputs = model(inputs)


        score = outputs[0][1]

        score.backward()

        gradients = inputs.grad
        

        total_gradients += gradients.reshape((6585, 128))

df_gradients = pd.DataFrame(total_gradients.detach().cpu().numpy())

df_gradients.to_csv('gradients.csv', index=False)


