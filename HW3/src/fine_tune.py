import numpy as np
import os
import pandas as pd
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models import resnet18
import glob
from tqdm import tqdm


class FinetuneDataset(Dataset):
    def __init__(self):
        self.data = np.load('./release/test.pkl', allow_pickle=True)
        self.task = None

    def __len__(self):
        return len(self.data['sup_images'][self.task])
    
    def __getitem__(self, idx):
        return torch.tensor(self.data['sup_images'][self.task][idx]) / 255, self.data['sup_labels'][self.task][idx]

class TestDataset(Dataset):
    def __init__(self):
        self.data = np.load('./release/test.pkl', allow_pickle=True)
        self.task = None

    def __len__(self):
        return len(self.data['qry_images'][self.task])

    def __getitem__(self, idx):
        return torch.tensor(self.data['qry_images'][self.task][idx]) / 255


class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, 80)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


def fine_tune(model, train_loader, optimizer, criterion, epoch, max_epoch, device, task_num):
    
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True
    
    model.train()
    train_loss = []
    avg_loss = []
    train_acc = []
    # pbar = tqdm(train_loader, desc=f"[Task:{task_num} | Epoch: {epoch}/{max_epoch}]")
    for (data, target) in train_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        avg_loss.append(np.mean(train_loss))
        # pbar.set_postfix({'loss': avg_loss[-1]})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = (torch.argmax(output, dim=1) == target).sum().item()
        train_acc.append(correct / len(data))
    # print(f'Training Loss: {avg_loss[-1]:.6f} \tTraining Accuracy: {np.mean(train_acc):.6f}') 

def eval(model, test_loader, device):
    model.eval()
    test_pred = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device).squeeze(0)
            output = model(data)
            test_pred += (output.argmax(dim=1).cpu().tolist())
    
    return test_pred
   

def main():
    # fine tune
    print("Start fine tuning...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_epoch = 25
    total_pred = []
    fine_tune_dataset = FinetuneDataset()
    test_dataset = TestDataset()
    for i in tqdm(range(600)):
        model = PretrainModel().to(device)
        model.load_state_dict(torch.load('./pretrain_weight.pth'))
        model.resnet.fc = nn.Linear(512, 5).to(device)
        model.train()
        fine_tune_dataset.task = i
        fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=128, shuffle=True, num_workers=0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, max_epoch+1):
            fine_tune(model, fine_tune_loader, optimizer, criterion, epoch, max_epoch, device, i)

        # predict
        model.eval()
        test_dataset.task = i
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        total_pred  = total_pred + eval(model, test_loader, device)
    
    # output csv 
    df = pd.DataFrame({'Id':np.arange(0, len(total_pred)), 'Category':total_pred})
    df.to_csv('./submission.csv', index=False, header=True)

if __name__ == '__main__':
    main()