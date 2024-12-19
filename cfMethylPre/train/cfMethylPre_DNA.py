import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from datetime import datetime



torch.cuda.init()


device = torch.device("cuda")


df = pd.read_csv('../data/pre/DNA_sample_encode.csv')

metrics = {}
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx, :-1]
        y = self.data.iloc[idx, -1] - 1
        x = torch.tensor(x, dtype=torch.float32)
        x = x.reshape(1, 6585, 128)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1256)



def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y



for fold, (train_idx, val_idx) in enumerate(kf.split(df, df.iloc[:, -1])):

    model = models.resnet101(pretrained=True).to(device)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 91)
    
    transfer_model = model

    transfer_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.0001, momentum=0.9)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = MyDataset(train_df)
    val_dataset = MyDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    num_epochs = 250
    for epoch in range(num_epochs):
        running_loss = 0.0

        transfer_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = transfer_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        transfer_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_prob_outputs = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = transfer_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                prob_outputs = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                val_prob_outputs.extend(prob_outputs)

                val_preds += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                val_targets += labels.cpu().numpy().tolist()


            val_accuracy = accuracy_score(val_targets, val_preds)
            val_precision = precision_score(val_targets, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_targets, val_preds, average='weighted', zero_division=0)
            val_f1_score = f1_score(val_targets, val_preds, average='weighted', zero_division=0)

            val_auroc = roc_auc_score(val_targets, val_prob_outputs, multi_class='ovo', average='macro', labels = list(range(91)))

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print('Time: %s, Fold %d, Epoch %d, Train loss: %.6f, Val loss: %.6f, Val accuracy: %.6f, Val precision: %.6f, Val recall: %.6f, Val F1-score: %.6f, Val auroc: %.6f' % 
                        (current_time, fold+1, epoch+1, running_loss/len(train_loader), val_loss/len(val_loader), val_accuracy, val_precision, val_recall, val_f1_score, val_auroc))

        metrics_list = [(val_accuracy, val_precision, val_recall, val_f1_score)]
        if fold in metrics:
            metrics[fold].extend(metrics_list)
        else:
            metrics[fold] = metrics_list

    torch.save(transfer_model.state_dict(), '../model/all_dna_101_{}.pt'.format(fold+1))


with open('../model/all_dna_101.txt', 'w') as f:
    for fold, values in metrics.items():
        for i, (val_accuracy, val_precision, val_recall, val_f1_score, val_auroc) in enumerate(values):
            f.write('Fold %d, Epoch %d, Val accuracy: %.6f, Val precision: %.6f, Val recall: %.6f, Val F1-score: %.6f, Val auroc: %.6f\n' % (fold+1, i+1, val_accuracy, val_precision, val_recall, val_f1_score, val_auroc))
