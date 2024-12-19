import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm
import numpy as np
from datetime import datetime



device = torch.device("cuda")


label_file = 'train_GSE214344_640_128.txt'
output_file = '5fd_mcc_results.txt'
with open(label_file, 'r') as f:
    lines = f.readlines()
    data = [line.strip().split() for line in lines]
    filenames = [item[0] for item in data]
    labels = [int(item[1]) for item in data]


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


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1256)


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return x, y

labelX=labels

for fold, (train_idx, val_idx) in enumerate(kf.split(filenames, labels)):
    
    train_filenames = [filenames[i] for i in train_idx]
    train_labels = [labelX[i] for i in train_idx]
    val_filenames = [filenames[i] for i in val_idx]
    val_labels = [labelX[i] for i in val_idx]
    train_dataset = MyDataset(train_filenames, train_labels)
    val_dataset = MyDataset(val_filenames, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)


    model = models.resnet101(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 91)
    model.load_state_dict(torch.load('../model/all_dna_101_1.pt'))
    
    transfer_model = model

    for param in transfer_model.parameters():
        param.requires_grad = False


    for param in transfer_model.layer3.parameters():
        param.requires_grad = True

    for param in transfer_model.layer4.parameters():
        param.requires_grad = True

    for param in transfer_model.fc.parameters():
        param.requires_grad = True

    transfer_model.fc = nn.Linear(transfer_model.fc.in_features, 10)
    transfer_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.000001)

    num_epochs = 250
    pbar = tqdm(range(num_epochs))
    
    for epoch in pbar:
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

                val_preds += torch.argmax(outputs,dim=1).cpu().numpy().tolist()
                val_targets += labels.cpu().numpy().tolist()


            val_accuracy = accuracy_score(val_targets,val_preds)
            val_precision=precision_score(val_targets,val_preds,
                                          average='weighted',zero_division=0)
            val_recall=recall_score(val_targets,val_preds,
                                    average='weighted',zero_division=0)
            val_f1_score=f1_score(val_targets,val_preds,
                                  average='weighted',zero_division=0)


            val_auroc=roc_auc_score(val_targets,val_prob_outputs,
                                    multi_class='ovo',average='macro',
                                    labels=list(range(10)))
            mcc_score = matthews_corrcoef(val_targets, val_preds)


            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            

            pbar.set_postfix(val_accuracy=val_accuracy, 
                             val_precision=val_precision, 
                             val_recall=val_recall, 
                             val_f1_score=val_f1_score, 
                             val_auroc=val_auroc,
                             mcc_score=mcc_score)
            
            with open(output_file, 'a') as f:
                f.write('Time: %s, Fold %d, Epoch %d, Train loss: %.6f, Val loss: %.6f, Val accuracy: %.6f, Val precision: %.6f, Val recall: %.6f, Val F1-score: %.6f, Val auroc: %.6f, MCC: %.6f\n' % 
                        (current_time,fold+1,epoch+1,
                         running_loss/len(train_loader),
                         val_loss/len(val_loader),
                         val_accuracy,val_precision,
                         val_recall,val_f1_score,val_auroc,mcc_score))

    torch.save(transfer_model.state_dict(),
               '../model/5fd_all_{}.pt'.format(fold+1))
